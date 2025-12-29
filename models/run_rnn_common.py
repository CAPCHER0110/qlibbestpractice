import os
import pickle
import json
import redis
import pandas as pd
from datetime import datetime, timedelta
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.data.dataset import TSDatasetH  
import importlib
from models.backtest_utils import run_backtest_analysis
import copy
from models.stock_utils import stock_mapper
from tabulate import tabulate

# =============================================================================
# 辅助函数：动态加载类
# =============================================================================
def get_dataset_class(class_name, module_path):
    """
    根据配置文件动态加载类
    例如: class_name="MTSDatasetH", module_path="qlib.contrib.data.dataset"
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ 动态加载数据集类失败: {module_path}.{class_name}")
        raise e
    
# =============================================================================
# 哑巴写入器 (用于欺骗 TRA 模型)
# =============================================================================
class DummyWriter:
    """这是一个假的 Tensorboard Writer，什么都不做，只为防止报错"""
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass
    def flush(self):
        pass

def run_train(config, model_name):
    """通用的 RNN 类模型训练逻辑"""
    print(f">>> [{model_name}] 进入训练流程...")
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    # Experiment Name 动态命名
    with R.start(experiment_name=f"{model_name.lower()}_production_run"):
        print("    构建数据集 (Time Series)...")
        dataset_config = config['task']['dataset']
        # 注意：Config 文件里的 class 也需要确保是 TSDatasetH
        dataset = init_instance_by_config(dataset_config)
        
        print(f"    初始化 {model_name} 模型...")
        model_config = config['task']['model']
        model = init_instance_by_config(model_config)
        
        print("    开始 Training...")
        model.fit(dataset)
        
        # 保存路径
        model_path = f"artifacts/{model_name.lower()}_model_latest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ 模型已保存至: {model_path}")

        # 简单验证
        try:
            pred = model.predict(dataset)
            pred.to_pickle(f"predictions/{model_name.lower()}_test_pred.pkl")
        except Exception as e:
            print(f"⚠️ 验证集预测失败 (可能是内存不足或数据切片问题): {e}")

# =============================================================================
# 推理逻辑 (run_predict) 
# =============================================================================
def run_predict(config, model_name):
    print(f">>> [{model_name}] 进入每日推理流程...")
    
    model_path = f"artifacts/{model_name.lower()}_model_latest.pkl"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}。请先运行 --mode train")
        return

    # 1. 加载模型
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # -------------------------------------------------------------------------
    # 【核心修复】给模型打补丁，注入假的 writer
    # -------------------------------------------------------------------------
    if not hasattr(model, '_writer') or model._writer is None:
        # print("    🔧 检测到模型缺少 _writer，正在注入 DummyWriter...")
        model._writer = DummyWriter()
    
    # 有些旧版 TRA 可能还需要 global_step
    if not hasattr(model, 'global_step'):
        model.global_step = 0
    # -------------------------------------------------------------------------

    # 2. 准备时间窗口 (Rolling Fit)
    dataset_conf_origin = config['task']['dataset']
    # TRA 也是 step_len
    seq_len = dataset_conf_origin['kwargs'].get('step_len', 40)
    
    today = datetime.now().strftime("%Y-%m-%d")
    lookback_start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    print(f"    数据采样: {lookback_start} ~ {today} (Rolling Fit)")

    # 3. 动态配置 Handler
    # ... (以下代码保持之前的逻辑不变) ...
    handler_config = copy.deepcopy(dataset_conf_origin['kwargs']['handler'])
    handler_config['kwargs']['start_time'] = lookback_start
    handler_config['kwargs']['end_time'] = today
    handler_config['kwargs']['fit_start_time'] = lookback_start
    handler_config['kwargs']['fit_end_time'] = today
    
    print(f"    正在初始化 DataHandler (Alpha158)...")
    try:
        dh = init_instance_by_config(handler_config)
    except Exception as e:
        print(f"❌ DataHandler 初始化失败: {e}")
        return

    # 4. 实例化 Dataset (动态加载)
    ds_class_name = dataset_conf_origin['class']
    ds_module_path = dataset_conf_origin['module_path']
    
    print(f"    正在初始化数据集: {ds_class_name}...")
    DatasetClass = get_dataset_class(ds_class_name, ds_module_path)
    
    ds_kwargs = copy.deepcopy(dataset_conf_origin['kwargs'])
    ds_kwargs['handler'] = dh
    ds_kwargs['segments'] = {"test": [lookback_start, today]}
    
    dataset = DatasetClass(**ds_kwargs)

    # 5. 推理
    print("    正在执行预测...")
    try:
        pred_score = model.predict(dataset)
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        print(f"    Dataset Type: {type(dataset)}")
        # 打印详细错误堆栈，方便排查其他问题
        import traceback
        traceback.print_exc()
        return
    
    if pred_score.empty:
        print("⚠️ 预测结果为空")
        return

    last_date = pred_score.index.get_level_values("datetime").max()
    print(f"    最新有效信号日期: {last_date}")
    
    # todays_pred = pred_score.loc[last_date].sort_values(ascending=False)
    
    # print(f"[{last_date}] Top 10 ({model_name}):")
    # 获取当天的预测数据
    daily_pred = pred_score.loc[last_date]
    
    # =========================================================================
    # 【核心修复】兼容 DataFrame (TRA) 和 Series (LSTM)
    # =========================================================================
    if isinstance(daily_pred, pd.DataFrame):
        # 如果是 DataFrame，取第一列作为排序依据（通常是 'score'）
        col_name = daily_pred.columns[0]
        # 排序，并提取为 Series，以保证后续逻辑统一
        todays_pred = daily_pred.sort_values(by=col_name, ascending=False)[col_name]
    else:
        # 如果是 Series，直接排序
        todays_pred = daily_pred.sort_values(ascending=False)
    # =========================================================================
    
    top_df = todays_pred.head(10).to_frame(name='score')
    top_df['name'] = top_df.index.map(stock_mapper.get_name)
    top_df = top_df[['name', 'score']]
    
    print(tabulate(top_df, headers=['代码', '股票名称', '预测得分'], tablefmt='psql', showindex=True))
    print("-" * 35)

    _push_to_redis(todays_pred, last_date, model_name)

def _push_to_redis(df, date_obj, model_name):
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
        target_key = f"TARGET_{model_name.upper()}_{date_obj.strftime('%Y-%m-%d')}"
        
        targets = {}
        
        # 【核心修复】兼容 Series (预测分) 和 DataFrame (带详情)
        if isinstance(df, pd.Series):
            # Series 使用 items() 迭代: (index, value)
            iterator = df.head(50).items()
        else:
            # DataFrame 使用 iterrows() 迭代: (index, Series)
            iterator = df.head(50).iterrows()

        for instrument, data in iterator:
            # 如果 data 是 Series (DataFrame的一行)，取 score 列；如果是数值 (Series的值)，直接用
            # 这里不需要用到分数本身做逻辑，只是为了循环
            
            if instrument.startswith('SH'): vt = f"{instrument[2:]}.SSE"
            elif instrument.startswith('SZ'): vt = f"{instrument[2:]}.SZSE"
            else: vt = instrument
            
            targets[vt] = 1000  # 默认目标持仓数量
            
        r.set(target_key, json.dumps(targets))
        print(f"✅ Redis 推送成功: Key={target_key}")
    except Exception as e:
        print(f"⚠️ Redis 推送失败: {e}")

def run_backtest(config, model_name):
    print(f">>> [{model_name}] 进入回测分析模式...")
    
    # 1. 确定预测文件路径
    # 我们回测的是之前训练时生成的“测试集预测结果”
    pred_path = f"predictions/{model_name.lower()}_test_pred.pkl"
    
    if not os.path.exists(pred_path):
        print(f"❌ 找不到预测文件: {pred_path}")
        print("   请先运行 --mode train 生成测试集预测结果。")
        return

    # 2. 加载预测数据
    print(f"    加载预测数据: {pred_path}")
    pred_df = pd.read_pickle(pred_path)
    
    # 3. 执行回测
    # 输出文件前缀: predictions/lstm
    run_backtest_analysis(
        pred_df, 
        output_prefix=f"predictions/{model_name.lower()}",
        topk=50, 
        benchmark=config.get('benchmark', 'sh000300')
    )
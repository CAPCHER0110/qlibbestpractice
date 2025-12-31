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
from qlib.data import D

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
# 推理逻辑 (run_predict) - 动态筛选版 & Redis 集成
# =============================================================================
def run_predict(config, model_name, target_pool="csi300"):
    print(f">>> [{model_name}] 进入每日推理流程...")
    print(f"    🎯 预测目标池: {target_pool}")
    
    model_path = f"artifacts/{model_name.lower()}_model_latest.pkl"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}。请先运行 --mode train")
        return

    # 1. 加载模型
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # -------------------------------------------------------------------------
    # 给模型打补丁 (DummyWriter)
    # -------------------------------------------------------------------------
    if not hasattr(model, '_writer') or model._writer is None:
        model._writer = DummyWriter()
    
    if not hasattr(model, 'global_step'):
        model.global_step = 0
    # -------------------------------------------------------------------------

    # =============== 必须加上这一行 ===============
    print(f"    🔧 [Local Fix] 强制重置 n_jobs=0 (原配置: {getattr(model, 'n_jobs', 'Unknown')})")
    model.n_jobs = 0  # <--- 强制让模型忘记服务器的高配，适应本地环境
    # ============================================

    # 2. 准备时间窗口
    dataset_conf_origin = config['task']['dataset']
    today = datetime.now().strftime("%Y-%m-%d")
    lookback_start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    print(f"    数据采样: {lookback_start} ~ {today} (Rolling Fit)")

    # 3. 动态配置 Handler
    # 这里我们深拷贝一份配置，以免修改了全局 Config
    handler_config = copy.deepcopy(dataset_conf_origin['kwargs']['handler'])
    
    # =========================================================================
    # 动态覆盖 instruments
    # =========================================================================
    if target_pool:
        handler_config['kwargs']['instruments'] = target_pool
        print(f"    🔄 已将数据加载范围锁定为: {target_pool}")
    # =========================================================================

    handler_config['kwargs']['start_time'] = lookback_start
    handler_config['kwargs']['end_time'] = today
    handler_config['kwargs']['fit_start_time'] = lookback_start
    handler_config['kwargs']['fit_end_time'] = today
    
    print(f"    正在初始化 DataHandler...")
    try:
        dh = init_instance_by_config(handler_config)
    except Exception as e:
        print(f"❌ DataHandler 初始化失败: {e}")
        return

    # 4. 实例化 Dataset
    ds_class_name = dataset_conf_origin['class']
    ds_module_path = dataset_conf_origin['module_path']
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
        return
    
    if pred_score.empty:
        print("⚠️ 预测结果为空")
        return

    last_date = pred_score.index.get_level_values("datetime").max()
    print(f"    最新有效信号日期: {last_date}")
    
    # 获取并排序当天的预测数据
    daily_pred = pred_score.loc[last_date]
    
    # 兼容 DataFrame/Series 排序
    if isinstance(daily_pred, pd.DataFrame):
        col_name = daily_pred.columns[0]
        todays_pred = daily_pred.sort_values(by=col_name, ascending=False)[col_name]
    else:
        todays_pred = daily_pred.sort_values(ascending=False)
    
    # 打印 Top 10
    top_df = todays_pred.head(10).to_frame(name='score')
    top_df['name'] = top_df.index.map(stock_mapper.get_name)
    top_df = top_df[['name', 'score']]
    
    print(f"[{last_date}] Top 10 ({model_name}) [Pool: {target_pool}]:")
    print(tabulate(top_df, headers=['代码', '股票名称', '预测得分'], tablefmt='psql', showindex=True))
    print("-" * 35)

    # 推送 Redis
    _push_to_redis(todays_pred, last_date, model_name)

def _push_to_redis(df, date_obj, model_name):
    """
    Redis 推送辅助函数
    """
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
            # 格式化代码
            if instrument.startswith('SH'): vt = f"{instrument[2:]}.SSE"
            elif instrument.startswith('SZ'): vt = f"{instrument[2:]}.SZSE"
            else: vt = instrument
            
            targets[vt] = 1000  # 默认目标持仓数量
            
        r.set(target_key, json.dumps(targets))
        print(f"✅ Redis 推送成功: Key={target_key}")
    except Exception as e:
        print(f"⚠️ Redis 推送失败: {e}")


# =============================================================================
# 回测逻辑 (run_backtest) - 稳健过滤版 (Robust Filter)
# =============================================================================
def run_backtest(config, model_name, target_pool="csi300"):
    print(f">>> [{model_name}] 进入回测分析模式...")
    print(f"    🎯 回测目标池: {target_pool}")
    
    # 1. 确定预测文件路径
    pred_path = f"predictions/{model_name.lower()}_test_pred.pkl"
    
    if not os.path.exists(pred_path):
        print(f"❌ 找不到预测文件: {pred_path}")
        print("    请先运行 --mode train 生成测试集预测结果。")
        return

    # 2. 加载预测数据
    print(f"    加载预测数据: {pred_path}")
    pred_df = pd.read_pickle(pred_path)
    
    # =========================================================================
    # 【核心新增】稳健过滤逻辑 (List-based Filter)
    # =========================================================================
    if target_pool and target_pool != 'all':
        print(f"    🔄 正在过滤非 {target_pool} 成分股...")
        original_count = len(pred_df)
        
        try:
            start_time = pred_df.index.get_level_values("datetime").min()
            end_time = pred_df.index.get_level_values("datetime").max()
            
            # 1. 获取白名单 (as_list=True 忽略时间对齐，只看代码)
            valid_instruments = D.list_instruments(
                D.instruments(target_pool), 
                start_time=start_time, 
                end_time=end_time, 
                as_list=True
            )
            
            print(f"    📊 {target_pool} 有效成分股数量: {len(valid_instruments)}")
            
            if len(valid_instruments) == 0:
                print("    ⚠️ 警告: 成分股列表为空！请检查 instruments 文件。")
            else:
                # 2. 仅根据“股票代码”进行过滤
                # 这种方式极快，且不会因为某天行情缺失而丢数据
                pred_df = pred_df[pred_df.index.get_level_values("instrument").isin(valid_instruments)]
                
                print(f"    ✅ 过滤完成: {original_count} -> {len(pred_df)} 条记录")
                
        except Exception as e:
            print(f"⚠️ 过滤失败: {e}")
            print("    将继续使用原始数据回测...")
    else:
        print("    🚀 全市场回测 (target_pool=all)")

    # 3. 再次检查数据是否为空
    if pred_df.empty:
        print(f"❌ 错误: 过滤后数据为空！请检查 target_pool={target_pool} 是否正确，或时间范围是否匹配。")
        return

    # 4. 执行回测
    run_backtest_analysis(
        pred_df, 
        output_prefix=f"predictions/{model_name.lower()}_{target_pool}",
        topk=50, 
        benchmark=config.get('benchmark', 'SH000300')
    )
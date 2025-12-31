import os
import pickle
import pandas as pd
import copy
from datetime import datetime, timedelta
from tabulate import tabulate
import json
import redis

from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.data import D  # 【新增】必须导入 D 用于回测过滤
from qlib.data.dataset import DatasetH
from models.backtest_utils import run_backtest_analysis
from models.stock_utils import stock_mapper

# =============================================================================
# 辅助函数：滚动时间生成器
# =============================================================================
def get_rolling_tasks(start_date, end_date, train_window, step):
    """
    生成滚动训练的任务列表
    返回: [(train_start, train_end, test_start, test_end), ...]
    """
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    
    tasks = []
    for i in range(train_window, len(calendar), step):
        test_start_idx = i
        test_end_idx = min(i + step - 1, len(calendar) - 1)
        
        test_start = calendar[test_start_idx]
        test_end = calendar[test_end_idx]
        
        train_start_idx = i - train_window
        train_end_idx = i - 1  
        
        train_start = calendar[train_start_idx]
        train_end = calendar[train_end_idx]
        
        tasks.append((train_start, train_end, test_start, test_end))
        
        if test_end_idx == len(calendar) - 1:
            break
            
    return tasks

# =============================================================================
# 训练逻辑 (支持 Rolling)
# =============================================================================
def run_train(config, model_name="LGBM"):
    print(f">>> [{model_name}] 进入训练流程...")
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    # 1. 检查是否开启滚动
    rolling_conf = config.get('rolling', {})
    enable_rolling = rolling_conf.get('enable', False)

    if not enable_rolling:
        print("ℹ️  模式: 静态训练 (Static Training)")
        _train_static(config)
    else:
        print("🔄 模式: 滚动训练 (Rolling Retraining)")
        _train_rolling(config)

def _train_static(config):
    """静态训练逻辑"""
    with R.start(experiment_name="lgbm_static"):
        dataset = init_instance_by_config(config['task']['dataset'])
        model = init_instance_by_config(config['task']['model'])
        
        print("    开始静态训练...")
        model.fit(dataset)
        
        # 保存
        with open("artifacts/lgbm_model_latest.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # 预测
        pred = model.predict(dataset)
        pred.to_pickle("predictions/lgbm_test_pred.pkl")
        print("✅ 静态训练完成。")

def _train_rolling(config):
    """滚动训练核心逻辑"""
    step = config['rolling']['step']
    train_window = config['rolling']['train_window']
    test_range = config['task']['dataset']['kwargs']['segments']['test']
    
    handler_start = config['data_handler_config']['start_time']
    handler_end = config['data_handler_config']['end_time']
    
    print(f"    计算滚动任务 (Step={step}, Window={train_window})...")
    
    full_calendar = D.calendar(start_time=handler_start, end_time=handler_end)
    test_start_date = pd.Timestamp(test_range[0])
    
    try:
        start_idx = full_calendar.tolist().index(test_start_date)
    except ValueError:
        start_idx = full_calendar.searchsorted(test_start_date)

    all_preds = []
    
    dataset_conf = config['task']['dataset']
    print("    正在初始化 DataHandler (一次性加载)...")
    handler = init_instance_by_config(dataset_conf['kwargs']['handler'])

    current_idx = start_idx
    last_model = None
    
    while current_idx < len(full_calendar):
        test_end_idx = min(current_idx + step - 1, len(full_calendar) - 1)
        
        train_start_idx = current_idx - train_window
        train_end_idx = current_idx - 1
        
        if train_start_idx < 0:
            print("⚠️ 数据历史不足以支持第一个训练窗口，跳过。")
            current_idx += step
            continue

        train_start = full_calendar[train_start_idx]
        train_end = full_calendar[train_end_idx]
        test_start = full_calendar[current_idx]
        test_end = full_calendar[test_end_idx]
        
        print(f"    🔄 Rolling: Train[{train_start.date()} ~ {train_end.date()}] -> Test[{test_start.date()} ~ {test_end.date()}]")

        sub_dataset = DatasetH(
            handler=handler,
            segments={
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            }
        )
        
        model = init_instance_by_config(config['task']['model'])
        model.fit(sub_dataset)
        last_model = model 
        
        pred = model.predict(sub_dataset, segment='test')
        all_preds.append(pred)
        
        current_idx += step

    if all_preds:
        final_pred = pd.concat(all_preds)
        final_pred.to_pickle("predictions/lgbm_rolling_backtest.pkl")
        print(f"✅ 滚动回测完成，累计预测样本数: {len(final_pred)}")
    
    if last_model:
        with open("artifacts/lgbm_model_latest.pkl", "wb") as f:
            pickle.dump(last_model, f)
        print(f"💾 最新模型已保存 (训练截止至: {full_calendar[min(current_idx-1, len(full_calendar)-1)].date()})")


# =============================================================================
# 推理逻辑 (run_predict) - 增加 target_pool 支持 & Redis 推送
# =============================================================================
def run_predict(config, model_name="LGBM", target_pool="csi300"):
    print(f">>> [{model_name}] 进入每日推理流程...")
    print(f"    🎯 预测目标池: {target_pool}")

    model_path = "artifacts/lgbm_model_latest.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}。请先运行 --mode train")
        return

    # 1. 加载模型
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. 准备时间窗口
    today = datetime.now().strftime("%Y-%m-%d")
    # Alpha158 需要较长的历史数据来计算 Rolling(60) 等特征，建议保留 100 天缓冲
    lookback_start = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    print(f"    数据采样: {lookback_start} ~ {today}")

    # 3. 动态配置 Handler
    # 复制整个 handler 配置结构
    handler_config = copy.deepcopy(config['task']['dataset']['kwargs']['handler'])
    
    # =========================================================================
    # 【核心】动态覆盖 instruments (支持 csi300/csi500/all)
    # 这会显著加快推理速度，因为只加载需要的股票
    # =========================================================================
    if target_pool:
        handler_config['kwargs']['instruments'] = target_pool
        print(f"    🔄 已将数据加载范围锁定为: {target_pool}")
    
    handler_config['kwargs']['start_time'] = lookback_start
    handler_config['kwargs']['end_time'] = today
    
    # 移除 fit_time (推理阶段不需要)
    handler_config['kwargs'].pop('fit_start_time', None)
    handler_config['kwargs'].pop('fit_end_time', None)
    
    print("    正在初始化 DataHandler (Alpha158)...")
    try:
        dh = init_instance_by_config(handler_config)
    except Exception as e:
        print(f"❌ DataHandler 初始化失败: {e}")
        print(f"    💡 提示: 请检查本地数据是否包含 {target_pool} 的定义，或者数据是否已更新到今天。")
        return

    # 4. 实例化 Dataset
    ds = DatasetH(handler=dh, segments={"test": [lookback_start, today]})
    
    # 5. 推理
    print("    正在执行预测...")
    try:
        pred = model.predict(ds)
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return
    
    if not pred.empty:
        last_date = pred.index.get_level_values("datetime").max()
        print(f"    最新有效信号日期: {last_date}")
        
        # 获取当天的预测数据
        daily_pred = pred.loc[last_date]
        
        # 兼容 DataFrame/Series 排序 (稳健性处理)
        if isinstance(daily_pred, pd.DataFrame):
            col_name = daily_pred.columns[0]
            todays_pred = daily_pred.sort_values(by=col_name, ascending=False)[col_name]
        else:
            todays_pred = daily_pred.sort_values(ascending=False)
        
        print(f"[{last_date}] Top 10 (LGBM) [Pool: {target_pool}]:")
        
        # 结果可视化
        top_df = todays_pred.head(10).to_frame(name='score')
        top_df['name'] = top_df.index.map(stock_mapper.get_name)
        top_df = top_df[['name', 'score']]
        
        print(tabulate(top_df, headers=['代码', '股票名称', '预测得分'], tablefmt='psql', showindex=True))
        print("-" * 35)
        
        # 【新增】推送至 Redis
        _push_to_redis(todays_pred, last_date, model_name)
        
    else:
        print("⚠️ 预测结果为空")

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

# =============================================================================
# 回测逻辑 (run_backtest) - 增加 target_pool 过滤
# =============================================================================
# =============================================================================
# 回测逻辑 (run_backtest) - 稳健过滤版
# =============================================================================
def run_backtest(config, model_name="LGBM", target_pool="csi300"):
    print(f">>> [{model_name}] 进入回测分析模式...")
    print(f"    🎯 回测目标池: {target_pool}")
    
    # 路径检查
    rolling_path = "predictions/lgbm_rolling_backtest.pkl"
    static_path = "predictions/lgbm_test_pred.pkl"
    
    if os.path.exists(rolling_path):
        pred_path = rolling_path
        print("    检测到滚动回测数据 (Rolling Backtest)")
    elif os.path.exists(static_path):
        pred_path = static_path
        print("    检测到静态测试数据 (Static Test)")
    else:
        print("❌ 找不到预测文件。请先运行 --mode train")
        return

    print(f"    加载预测数据: {pred_path}")
    pred_df = pd.read_pickle(pred_path)
    
    # =========================================================================
    # 【核心修复】稳健的过滤逻辑 (List-based Filter)
    # =========================================================================
    if target_pool and target_pool != 'all':
        print(f"    🔄 正在过滤非 {target_pool} 成分股...")
        original_count = len(pred_df)
        
        try:
            start_time = pred_df.index.get_level_values("datetime").min()
            end_time = pred_df.index.get_level_values("datetime").max()
            
            # 1. 直接获取这段时间内 CSI300 的“成分股白名单” (List)
            # as_list=True 会返回所有曾经入选过的股票列表，非常稳健
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
                # 2. 仅根据“股票代码”进行过滤 (忽略日期的严格对齐)
                # 这种方式极快，且不会因为某天行情缺失而丢数据
                pred_df = pred_df[pred_df.index.get_level_values("instrument").isin(valid_instruments)]
                
                print(f"    ✅ 过滤完成: {original_count} -> {len(pred_df)} 条记录")
                
        except Exception as e:
            print(f"    ⚠️ 过滤失败: {e}")
            print("    将继续使用原始数据回测...")
            
    else:
        print("    🚀 全市场回测 (target_pool=all)")

    # 再次检查
    if pred_df.empty:
        print("❌ 错误: 过滤后数据为空！请检查 target_pool 是否正确。")
        return

    # 执行回测
    run_backtest_analysis(
        pred_df,
        output_prefix=f"predictions/lgbm_{target_pool}",
        topk=50,
        benchmark=config.get('benchmark', 'SH000300')
    )

# =============================================================================
# 统一接口 (兼容 Main.py)
# =============================================================================
def execute(config, mode, target_pool="csi300"):
    if mode == 'train':
        run_train(config)
    elif mode == 'predict':
        run_predict(config, target_pool=target_pool)
    elif mode == 'backtest':
        run_backtest(config, target_pool=target_pool)
    else:
        print(f"❌ 不支持的模式: {mode}")
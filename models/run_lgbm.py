import os
import pickle
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.data import D
from qlib.data.dataset import DatasetH
from models.backtest_utils import run_backtest_analysis
from datetime import datetime, timedelta
import copy
from tabulate import tabulate
from models.stock_utils import stock_mapper

# =============================================================================
# 辅助函数：滚动时间生成器
# =============================================================================
def get_rolling_tasks(start_date, end_date, train_window, step):
    """
    生成滚动训练的任务列表
    返回: [(train_start, train_end, test_start, test_end), ...]
    """
    # 获取全量交易日历
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    
    tasks = []
    # 从 train_window 开始，每隔 step 天滚动一次
    # i 是当前测试段的开始索引
    for i in range(train_window, len(calendar), step):
        # 确定测试段 (Test Segment)
        test_start_idx = i
        test_end_idx = min(i + step - 1, len(calendar) - 1)
        
        test_start = calendar[test_start_idx]
        test_end = calendar[test_end_idx]
        
        # 确定训练段 (Train Segment) = 测试段起点 - train_window
        train_start_idx = i - train_window
        train_end_idx = i - 1  # 训练到测试开始前一天
        
        train_start = calendar[train_start_idx]
        train_end = calendar[train_end_idx]
        
        tasks.append((train_start, train_end, test_start, test_end))
        
        if test_end_idx == len(calendar) - 1:
            break
            
    return tasks

# =============================================================================
# 训练逻辑 (支持 Rolling)
# =============================================================================
def _train(config):
    print(">>> [LGBM] 进入训练流程...")
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
    """旧的静态训练逻辑"""
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
    """
    滚动训练核心逻辑
    """
    # 1. 获取滚动参数
    step = config['rolling']['step']
    train_window = config['rolling']['train_window']
    
    # 使用 config 中定义的 'test' 段作为整个滚动的起止范围
    # 比如: 2017-01-01 到 2024-12-31
    test_range = config['task']['dataset']['kwargs']['segments']['test']
    # 为了保证第一次训练有数据，我们需要把 calendar 的起点往前推 train_window 天
    # 这里简化处理：我们假设数据源足够长，直接用 calendar 算
    # 获取需要覆盖的测试范围日历
    
    # 这里的 start_date 应该是整个大回测的开始时间
    # 我们需要找到第一段测试(2017-01-01) 对应的 训练开始时间(2016-xx-xx)
    # 简单起见，我们直接传入 config 中的数据起止时间
    handler_start = config['data_handler_config']['start_time']
    handler_end = config['data_handler_config']['end_time']
    
    print(f"    计算滚动任务 (Step={step}, Window={train_window})...")
    
    # 这里有一个小技巧：我们只在 test 段上进行滚动预测
    # 但 calendar 需要取更早的时间以包含训练数据
    # 为了方便，我们直接基于全量日历计算索引，然后只执行在 test_range 范围内的任务
    
    full_calendar = D.calendar(start_time=handler_start, end_time=handler_end)
    test_start_date = pd.Timestamp(test_range[0])
    
    # 找到 test_start 在日历中的索引
    try:
        start_idx = full_calendar.tolist().index(test_start_date)
    except ValueError:
        # 如果不是交易日，找最近的一个
        # 这里简化处理，直接用 searchsorted
        start_idx = full_calendar.searchsorted(test_start_date)

    all_preds = []
    
    # 准备 dataset 模板 (Handler 复用)
    dataset_conf = config['task']['dataset']
    # 实例化一个 Handler 对象 (比较耗时，只做一次)
    print("    正在初始化 DataHandler (一次性加载)...")
    handler = init_instance_by_config(dataset_conf['kwargs']['handler'])

    # 开始循环
    # 我们从 start_idx 开始作为第一个测试段的起点
    # 训练段就是 [start_idx - train_window, start_idx - 1]
    
    current_idx = start_idx
    last_model = None
    
    while current_idx < len(full_calendar):
        # 1. 确定时间窗口
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

        # 2. 动态切分 Dataset
        # 利用 DatasetH 的 segments 功能，直接传入 handler 对象，速度很快
        sub_dataset = DatasetH(
            handler=handler,
            segments={
                'train': (train_start, train_end),
                'test': (test_start, test_end) # 这里用 valid 还是 test 取决于你想不想验证，通常 rolling 用 test
            }
        )
        
        # 3. 初始化并训练模型
        model = init_instance_by_config(config['task']['model'])
        model.fit(sub_dataset)
        last_model = model # 暂存最新模型
        
        # 4. 预测当前段
        pred = model.predict(sub_dataset, segment='test')
        all_preds.append(pred)
        
        # 5. 步进
        current_idx += step

    # === 循环结束 ===
    
    # 1. 保存拼接后的回测结果
    if all_preds:
        final_pred = pd.concat(all_preds)
        final_pred.to_pickle("predictions/lgbm_rolling_backtest.pkl")
        print(f"✅ 滚动回测完成，累计预测样本数: {len(final_pred)}")
    
    # 2. 【关键】保存最后一个模型作为 Latest
    # 这样 predict 模式就会使用这个用“最新数据”训练出来的模型
    if last_model:
        with open("artifacts/lgbm_model_latest.pkl", "wb") as f:
            pickle.dump(last_model, f)
        print(f"💾 最新模型已保存 (训练截止至: {full_calendar[min(current_idx-1, len(full_calendar)-1)].date()})")


# =============================================================================
# 推理逻辑 (保持不变，或微调)
# =============================================================================
def _predict(config):
    print(">>> [LGBM] 进入每日推理流程...")
    model_path = "artifacts/lgbm_model_latest.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}。请先运行 --mode train")
        return

    # 1. 加载模型
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. 准备时间窗口
    today = datetime.now().strftime("%Y-%m-%d")
    # Alpha158 需要较长的历史数据来计算 Rolling(60) 等特征，建议 100 天缓冲
    lookback_start = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    print(f"    数据采样: {lookback_start} ~ {today}")

    # 3. 动态配置 Handler (使用 init_instance_by_config)
    # 【关键修改】复制整个 handler 配置结构 (包含 class, module_path, kwargs)
    handler_config = copy.deepcopy(config['task']['dataset']['kwargs']['handler'])
    
    # 覆盖起止时间
    handler_config['kwargs']['start_time'] = lookback_start
    handler_config['kwargs']['end_time'] = today
    
    # 移除 fit_time (推理阶段不需要)
    handler_config['kwargs'].pop('fit_start_time', None)
    handler_config['kwargs'].pop('fit_end_time', None)
    
    # 【核心修复】让 Qlib 自动去实例化 Alpha158，而不是手动调用 DataHandlerLP
    # 这样 Alpha158 会自动填充 data_loader，解决 AssertionError
    print("    正在初始化 DataHandler (Alpha158)...")
    dh = init_instance_by_config(handler_config)

    # 4. 实例化 Dataset
    ds = DatasetH(handler=dh, segments={"test": [lookback_start, today]})
    
    # 5. 推理
    print("    正在执行预测...")
    pred = model.predict(ds)
    
    if not pred.empty:
        last_date = pred.index.get_level_values("datetime").max()
        print(f"    最新有效信号日期: {last_date}")
        
        todays_pred = pred.loc[last_date].sort_values(ascending=False)
        # ==========================================
        # 【修改点】使用 stock_mapper 和 tabulate 显示
        # ==========================================
        print(f"[{last_date}] Top 10 (LGBM):")
        
        # 1. 转 DataFrame
        top5_df = todays_pred.head(10).to_frame(name='score')
        
        # 2. 映射名称
        top5_df['name'] = top5_df.index.map(stock_mapper.get_name)
        
        # 3. 整理列 (名称在前，分数在后)
        top5_df = top5_df[['name', 'score']]
        
        # 4. 打印漂亮表格
        print(tabulate(top5_df, headers=['代码', '股票名称', '预测得分'], tablefmt='psql', showindex=True))
        print("-" * 35)
        
        # _push_to_redis(todays_pred, last_date, "LGBM")
    else:
        print("⚠️ 预测结果为空")

def _backtest(config):
    print(">>> [LGBM] 进入回测分析模式...")
    
    # 优先寻找滚动回测的结果，如果没有则找静态的
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

    pred_df = pd.read_pickle(pred_path)
    
    run_backtest_analysis(
        pred_df,
        output_prefix="predictions/lgbm",
        topk=50,
        benchmark=config.get('benchmark', 'sh000300')
    )

# =============================================================================
# 统一接口
# =============================================================================
def execute(config, mode):
    if mode == 'train':
        _train(config)
    elif mode == 'predict':
        _predict(config)
    elif mode == 'backtest':
        _backtest(config)
    else:
        print(f"❌ 不支持的模式: {mode}")
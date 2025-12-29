import qlib
import yaml
import pandas as pd
import pickle
import os
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.backtest import backtest, executor

# 1. 加载配置
with open("config/workflow_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化
qlib.init(**config['qlib_init'])

# 滚动参数
ROLLING_STEP = config['rolling']['step']
train_start = "2018-01-01"
test_start = "2022-01-01"
end_date = "2023-12-31"

# 准备存储目录
os.makedirs("artifacts", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# === 核心：滚动训练 (Rolling Training) ===
# 模拟真实的时间流逝：不用未来数据训练
print(">>> 开始滚动训练...")

recorder = R.get_recorder(experiment_name="rolling_best_practice")
all_pred = []

# 这里简化逻辑，使用 Qlib 的 Rolling 发生器会更优雅，但为了代码可读性写个 Loop
# 在实际工程中，通常使用 `RollingGen`
from qlib.utils.time import Freq
from qlib.data import D

# 获取日历
calendar = D.calendar(start_time=test_start, end_time=end_date, freq='day')
# 按步长切分测试集
segments = [calendar[i:i+ROLLING_STEP] for i in range(0, len(calendar), ROLLING_STEP)]

for i, segment in enumerate(segments):
    seg_start, seg_end = segment[0], segment[-1]
    train_end = (seg_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Round {i}: Train [..{train_end}] -> Predict [{seg_start}..{seg_end}]")
    
    # 1. 动态更新数据集配置
    task = config['task'].copy()
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": config['data_handler_config'],
            "segments": {
                "train": [train_start, train_end],
                "valid": [train_end, train_end], # 简化：验证集
                "test":  [seg_start.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d")]
            }
        }
    }
    
    # 2. 训练模型
    model = init_instance_by_config(task['model'])
    dataset = init_instance_by_config(dataset_config)
    model.fit(dataset)
    
    # 3. [关键步骤] 保存模型快照 (Artifacts)
    # 我们保存最新的模型，用于实盘
    with open(f"artifacts/model_latest.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # 4. 预测
    pred = model.predict(dataset)
    all_pred.append(pred)

# === 信号合并 ===
full_pred = pd.concat(all_pred).sort_index()
full_pred.to_pickle("predictions/rolling_pred.pkl")
print(">>> 滚动训练完成，预测结果已保存。")

# === 回测分析 (Backtest) ===
print(">>> 开始回测...")

# 定义回测策略 (TopK)
STRATEGY_CONFIG = {
    "topk": 50,
    "n_drop": 5,
    "signal": full_pred, # 传入刚才生成的信号
}

# 运行回测
report_normal, positions = backtest(
    start_time=test_start, 
    end_time=end_date, 
    strategy_config=STRATEGY_CONFIG,
    account=100000000, 
    benchmark=config['benchmark'],
    exchange_kwargs={"limit_threshold": 0.095, "deal_price": "close"}
)

# 打印绩效
analysis = dict()
analysis['excess_return_without_cost'] = backtest_stats(report_normal, report_normal) # 简化的分析函数调用
print(report_normal)
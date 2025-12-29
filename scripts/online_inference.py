import qlib
import pickle
import yaml
import redis
import json
import pandas as pd
from datetime import datetime, timedelta
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# 1. 初始化
with open("config/workflow_config.yaml", "r") as f:
    config = yaml.safe_load(f)
qlib.init(**config['qlib_init'])

# 2. 确定时间窗口
# 只要最近 60 天的数据来计算因子 (足够算 20日均线了)
today = datetime.now().strftime("%Y-%m-%d")
start_lookback = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

print(f"[{today}] 开始在线推理...")

# 3. 准备推理数据集 (Inference Dataset)
# 这里我们只需要"test"段，不需要train
data_handler_config = config['data_handler_config']
data_handler_config['start_time'] = start_lookback
data_handler_config['end_time'] = today
data_handler_config['instruments'] = config['market'] # 或者指定具体的股票池

# 实例化 DataHandler (这一步会自动计算因子，非常耗时)
dh = DataHandlerLP(**data_handler_config['kwargs'])

# 4. 加载离线训练好的模型
print("加载模型...")
with open("artifacts/model_latest.pkl", "rb") as f:
    model = pickle.load(f)

# 5. 预测 (Inference)
# 我们需要构造一个 Dataset 对象喂给模型，但只包含今天的数据
ds = DatasetH(handler=dh, segments={"test": [today, today]})
pred_score = model.predict(ds)

# pred_score 索引是 (datetime, instrument)，列是 score
# 过滤掉 NaN，按分数排序
latest_scores = pred_score.loc[today].sort_values("score", ascending=False)
top_50 = latest_scores.head(50)

print(f"今日 Top 5:\n{top_50.head(5)}")

# 6. [关键] 推送到 Redis (供 Vn.py 使用)
r = redis.Redis(host='localhost', port=6379, db=0)
target_key = f"TARGET_{datetime.now().strftime('%Y-%m-%d')}"

# 构造目标仓位 JSON
targets = {}
for instrument, row in top_50.iterrows():
    # 格式转换: SH600519 -> 600519.SSE
    symbol = instrument[2:]
    exchange = "SSE" if instrument.startswith("SH") else "SZSE"
    vt_symbol = f"{symbol}.{exchange}"
    
    # 简单分配：平均持仓 (实际应根据 score 加权)
    targets[vt_symbol] = 1000 # 假设买 1000 股

r.set(target_key, json.dumps(targets))
print(f"✅ 信号已推送至 Redis: {target_key}")
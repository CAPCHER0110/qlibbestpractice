import pickle
import redis
import json
import pandas as pd
from datetime import datetime, timedelta
from qlib.data.dataset import MTSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

def run_inference_task(config):
    """
    加载模型并执行每日推理任务
    """
    print(">>> [Inference Task] 启动每日推理...")

    # 1. 路径定义 (建议从 config 读，这里为了简单先沿用硬编码)
    model_path = "artifacts/tra_model_latest.pkl"
    
    # 2. 加载模型
    print(f"    Loading model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ [Error] 找不到模型文件: {model_path}。请先执行训练模式。")
        return

    # 3. 准备时间窗口
    # 目标：预测“今天” (假设今天是交易日，或者是为了生成明天的信号)
    # 注意：实际生产中，通常是在 T 日晚上或 T+1 日早上，拿着 T 日的数据预测 T+1
    today = datetime.now().strftime("%Y-%m-%d")
    # 缓冲 60 天
    lookback_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"    Data Range: {lookback_start} ~ {today}")

    # 4. 准备数据 Handler
    # 复用 yaml 配置里的 handler 参数，确保和训练时特征一致
    handler_config = config['task']['dataset']['kwargs']['handler']
    handler_config['kwargs']['start_time'] = lookback_start
    handler_config['kwargs']['end_time'] = today

    # 实例化 Handler
    dh = DataHandlerLP(**handler_config['kwargs'])

    # 5. 实例化 Dataset
    # 使用训练时的 dataset 配置，但只取最近的一段
    dataset_config = config['task']['dataset']
    ts_dataset = MTSDatasetH(
        handler=dh,
        seq_len=dataset_config['kwargs']['seq_len'],
        segments={"test": [lookback_start, today]},
        step_len=dataset_config['kwargs'].get('step_len', 1)
    )

    # 6. 推理
    print("    Running prediction...")
    # pred_score index is (datetime, instrument), value is score
    pred_score = model.predict(ts_dataset)

    # 7. 提取今日信号并推送
    # 这里需要处理一下日期，Qlib 的数据通常包含“收盘时间”。
    # 如果今天是周末，或者数据还没更新，pred_score 里可能没有 today 的数据。
    # 我们尝试取 pred_score 的最后一天。
    if pred_score.empty:
        print("⚠️ [Warn] 预测结果为空，可能数据源未包含有效特征。")
        return

    last_available_date = pred_score.index.get_level_values("datetime").max()
    print(f"    Latest available prediction date: {last_available_date}")

    # 获取最后一日的预测
    todays_pred = pred_score.loc[last_available_date].sort_values("score", ascending=False)
    
    print(f"[{last_available_date}] Top 5 Predictions:")
    print(todays_pred.head(5))

    # 8. 推送 Redis
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        target_key = f"TARGET_{last_available_date.strftime('%Y-%m-%d')}"
        
        targets = {}
        for instrument, row in todays_pred.head(50).iterrows():
            # 格式转换 SH600000 -> 600000.SSE
            if instrument.startswith('SH'):
                vt_symbol = f"{instrument[2:]}.SSE"
            elif instrument.startswith('SZ'):
                vt_symbol = f"{instrument[2:]}.SZSE"
            else:
                vt_symbol = instrument
            
            targets[vt_symbol] = 1000 
            
        r.set(target_key, json.dumps(targets))
        print(f"✅ 信号已推送 Redis: Key={target_key}, Count={len(targets)}")
        
    except Exception as e:
        print(f"❌ [Error] Redis 推送失败: {e}")
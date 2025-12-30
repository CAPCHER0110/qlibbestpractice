# models/run_tra.py

# 直接调用我们封装好的通用逻辑
from models.run_rnn_common import run_train, run_predict, run_backtest

def execute(config, mode, target_pool):
    """
    TRA 模型执行入口
    (现在它只是一个薄薄的包装器，底层逻辑复用 run_rnn_common)
    """
    # 这里定义 model_name="TRA"
    # 通用逻辑会自动生成:
    # - artifacts/tra_model_latest.pkl
    # - predictions/tra_test_pred.pkl
    # - Redis Key: TARGET_TRA_2025-xx-xx
    
    model_name = "TRA"
    
    if mode == 'train':
        run_train(config, model_name=model_name)
    elif mode == 'predict':
        run_predict(config, model_name=model_name, target_pool=target_pool)
    elif mode == 'backtest': 
        run_backtest(config, model_name=model_name, target_pool=target_pool)
    else:
        print(f"❌ [TRA] 不支持的模式: {mode}")
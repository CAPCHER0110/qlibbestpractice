# 直接复用之前为 LSTM/ALSTM 编写的通用逻辑
# 如果你没有创建 run_rnn_common.py，请回顾上一条回答创建它
from models.run_rnn_common import run_train, run_predict, run_backtest

def execute(config, mode):
    """
    Transformer 模型的执行入口
    """
    # 指定模型名称为 "Transformer"，通用逻辑会自动处理日志和 Redis Key
    model_name = "Transformer"
    
    if mode == 'train':
        run_train(config, model_name=model_name)
    elif mode == 'predict':
        run_predict(config, model_name=model_name)
    elif mode == 'backtest': 
        run_backtest(config, model_name="ALSTM")
    else:
        print(f"❌ [Transformer] 不支持的模式: {mode}")
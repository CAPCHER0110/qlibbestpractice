from models.run_rnn_common import run_train, run_predict, run_backtest

def execute(config, mode):
    if mode == 'train':
        run_train(config, model_name="LSTM")
    elif mode == 'predict':
        run_predict(config, model_name="LSTM")
    elif mode == 'backtest': 
        run_backtest(config, model_name="LSTM")
    else:
        print(f"❌ 不支持的模式: {mode}")
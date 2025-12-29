from models.run_rnn_common import run_train, run_predict, run_backtest

def execute(config, mode):
    model_name = "LSTM"
    if mode == 'train':
        run_train(config, model_name=model_name)
    elif mode == 'predict':
        run_predict(config, model_name=model_name)
    elif mode == 'backtest': 
        run_backtest(config, model_name=model_name)
    else:
        print(f"❌ 不支持的模式: {mode}")
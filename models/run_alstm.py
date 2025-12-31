from models.run_rnn_common import run_train, run_predict, run_backtest

def execute(config, mode, target_pool):
    model_name = "ALSTM"
    if mode == 'train':
        run_train(config, model_name=model_name)
    elif mode == 'predict':
        run_predict(config, model_name=model_name, target_pool=target_pool)
    elif mode == 'backtest': 
        run_backtest(config, model_name=model_name, target_pool=target_pool)
    else:
        print(f"❌ 不支持的模式: {mode}")
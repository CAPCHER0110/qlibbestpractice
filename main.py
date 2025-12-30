import sys
import argparse
import yaml
import qlib
from pathlib import Path

# 路径修复
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.prepare_data import ensure_qlib_data

# === 导入我们的 TRA 模块 ===
# 假设 models 包下有 run_tra.py
from models import run_tra
from models import run_lgbm
from models import run_lstm
from models import run_alstm
from models import run_transformer

# === 模型注册表 ===
MODEL_DISPATCHER = {
    "tra": run_tra.execute,   # 注册 TRA
    "lgbm": run_lgbm.execute,
    "lstm": run_lstm.execute,   # <--- 注册 LSTM
    "alstm": run_alstm.execute, # <--- 注册 ALSTM
    "transformer": run_transformer.execute, # <--- 新增这一行注册
}

def main():
    parser = argparse.ArgumentParser(description="Qlib 量化交易系统")
    parser.add_argument('--model', type=str, default='lgbm', choices=MODEL_DISPATCHER.keys(), help='模型名称')
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict', 'backtest'], help='运行模式')
    # =========================================================================
    # 增加 filter 参数，默认为 csi300
    # =========================================================================
    parser.add_argument("--filter", type=str, default="csi300", help="预测/回测的目标池 (csi300/csi500/all)")
    args = parser.parse_args()

    # ... (中间的环境初始化代码，保持不变: check paths, ensure_data, qlib.init) ...
    # 为了节省篇幅，这里简写，请保留你之前完整的 main.py 逻辑
    # 只要改动 config_path 的获取逻辑和最后的分发逻辑即可
    
    local_data_dir = PROJECT_ROOT / "data" / "cn_data"
    
    # 动态加载对应模型的配置
    # 比如 tra -> config/tra_config.yaml
    config_path = PROJECT_ROOT / "config" / f"{args.model}_config.yaml"
    
    # 1. 数据检查
    # 如果是 predict 模式，只允许数据过期 1 天；如果是 train，宽容一点也没事
    ensure_qlib_data(max_age_days=1 if args.mode == 'predict' else 7)
    
    # 2. Qlib 初始化
    qlib.init(provider_uri=str(local_data_dir), region="cn")
    
    # 3. 读取配置
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "qlib_init" in config:
        del config["qlib_init"]

    # 4. 任务分发
    runner = MODEL_DISPATCHER.get(args.model)
    if runner:
        print(f"\n🚀 启动任务: Model={args.model.upper()}, Mode={args.mode.upper()}")
        # 调用 models/run_tra.py 里的 execute 函数
        runner(config, mode=args.mode, target_pool=args.filter)
    else:
        print(f"❌ 未注册的模型: {args.model}")

if __name__ == "__main__":
    main()
import os
import pickle
from qlib.utils import init_instance_by_config
from qlib.workflow import R

# 【移除】不再需要 import yaml，配置由外部传入
# 【移除】不再需要 import qlib (因为不再调用 qlib.init)

def run_tra_task(config):
    """
    执行 TRA 模型的训练与预测任务
    :param config: 已经在 main.py 里读取好的字典配置
    """
    
    # 【移除】这里不需要 qlib.init 了，main.py 已经做过了
    # qlib.init(**config['qlib_init']) 
    
    # 1. 准备目录 (确保 artifacts 和 predictions 目录存在)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    print(">>> [TRA Task] 开始构建数据和模型...")

    # 2. 启动实验记录
    # experiment_name 可以写死，也可以从 config 里读，这里保持原样
    with R.start(experiment_name="tra_production_run"):
        
        # === 数据集构建 ===
        dataset_config = config['task']['dataset']
        dataset = init_instance_by_config(dataset_config)
        
        # === 模型初始化 ===
        model_config = config['task']['model']
        model = init_instance_by_config(model_config)
        
        print(">>> [TRA Task] 开始 Training (建议去喝杯咖啡)...")
        model.fit(dataset)
        
        # === 保存模型 ===
        model_path = "artifacts/tra_model_latest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)     
        print(f">>> [TRA Task] 模型已保存至: {model_path}")

        # === 生成测试集预测 ===
        print(">>> [TRA Task] 生成测试集预测信号...")
        pred = model.predict(dataset)
        
        # 保存预测结果
        pred_path = "predictions/tra_test_pred.pkl"
        pred.to_pickle(pred_path)
        print(f">>> [TRA Task] 预测结果已保存至: {pred_path}")
        print(pred.head())
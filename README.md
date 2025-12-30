# Qlib Quantitative Trading Platform

基于 Microsoft Qlib 构建的生产级量化交易平台。本项目采用**策略模式 (Strategy Pattern)** 架构，实现了模型训练与每日推理的解耦，支持多模型（Deep Learning & GBDT）的统一管理与自动化运维。

## ✨ 核心特性

* **多模型支持**：统一接口支持 TRA, LightGBM, LSTM, ALSTM, Transformer 等多种 SOTA 模型。
* **零配置启动**：内置 `Infrastructure as Code` 逻辑，自动检测、下载并校验 Qlib 数据（A股），无需手动配置环境。
* **统一调度**：单一入口 (`main.py`)，通过命令行参数灵活切换模型与运行模式。
* **生产级推理**：
* 支持每日增量数据的自动拉取与清洗。
* 支持 LightGBM 的 **滚动训练 (Rolling Retraining)** 机制。
* 推理结果自动推送到 Redis，便于下游实盘系统对接。


* **高度可扩展**：基于 `run_rnn_common` 的通用模版，新增深度学习模型仅需添加配置文件和注册表项。

## 📂 项目结构

```text
.
├── main.py                  # [入口] 统一任务调度器 (Dispatcher)
├── config/                  # [配置] 各模型的 YAML 配置文件
│   ├── tra_config.yaml
│   ├── lgbm_config.yaml
│   ├── lstm_config.yaml
│   ├── alstm_config.yaml
│   └── transformer_config.yaml
├── models/                  # [模型] 业务逻辑实现
│   ├── __init__.py          # 模型注册表导出
│   ├── run_rnn_common.py    # 通用深度学习执行器 (TRA/LSTM/ALSTM/Transformer)
│   ├── run_lgbm.py          # LightGBM 执行器 (支持滚动训练)
│   ├── run_tra.py           # TRA 入口包装
│   ├── run_lstm.py          # LSTM 入口包装
│   └── ...
├── scripts/                 # [工具] 基础设施脚本
│   └── prepare_data.py      # 自动化数据下载与更新脚本
├── data/                    # [数据] 本地数据存储 (自动生成，已加 .gitignore)
│   └── cn_data/             # Qlib 二进制数据
├── artifacts/               # [产物] 训练好的模型文件 (*.pkl)
└── predictions/             # [产物] 预测结果与回测数据

```

## 🚀 快速开始

### 1. 环境准备

确保安装了 Python 3.8+ 及相关依赖：

```bash
pip install pyqlib torch pandas numpy redis pyyaml
# 建议根据显卡安装对应的 PyTorch 版本

```

### 2. 训练模型 (Training)

首次运行会自动下载约几百兆的 Qlib 基础数据（A股），请保持网络通畅。

**训练 TRA 模型：**

```bash
python main.py --model tra --mode train

```

**训练 LightGBM 模型 (支持滚动更新)：**

```bash
python main.py --model lgbm --mode train

```

**训练 Transformer 模型：**

```bash
python main.py --model transformer --mode train

```

### 3. 每日推理 (Daily Inference)

此模式适合配置在 Crontab 中每日盘前运行。系统会自动检查数据时效性，如果数据过期（>1天），会自动增量更新。

```bash
python main.py --model tra --mode predict

```

推理成功后，Top 50 的持仓信号将被推送到 Redis：

* **Key**: `TARGET_<MODEL>_<DATE>` (例如 `TARGET_TRA_2025-12-29`)
* **Value**: JSON 格式的持仓目标 `{ "600519.SSE": 1000, ... }`

## 🛠️ 模型与配置说明

| 模型名称 | 命令行参数 `--model` | 配置文件 | 说明 |
| --- | --- | --- | --- |
| **LightGBM** | `lgbm` | `lgbm_config.yaml` | GBDT 树模型。**支持 Rolling Retraining (滚动重训)**，抗噪性强。 |
| **LSTM** | `lstm` | `lstm_config.yaml` | 经典 RNN 模型，深度学习基准。 |
| **ALSTM** | `alstm` | `alstm_config.yaml` | 引入 Attention 机制的 LSTM。 |
| **Transformer** | `transformer` | `transformer_config.yaml` | 基于 Self-Attention 的时序模型。 |
| **TRA** | `tra` | `tra_config.yaml` | Temporal Routing Adaptor，擅长捕捉长期依赖。 |

### 关于 LightGBM 的滚动训练

在 `config/lgbm_config.yaml` 中，可以通过 `rolling` 字段控制是否开启滚动训练：

```yaml
rolling:
    enable: True
    step: 20           # 每隔 20 个交易日重新训练一次
    train_window: 242  # 每次回看过去一年的数据

```

开启后，执行 `--mode train` 将进行全量滚动回测，并保存最后一个时间窗口的模型用于每日预测。

## ⚙️ 进阶开发指南

### 如何添加新模型？

得益于策略模式，添加新模型非常简单：

1. **准备配置**：在 `config/` 下创建 `new_model_config.yaml`。
2. **编写逻辑**：
* 如果是标准 RNN/Transformer 类：在 `models/` 下新建 `run_new_model.py`，直接调用 `run_rnn_common`。
* 如果是特殊模型：仿照 `run_lgbm.py` 编写独立的 `execute` 函数。


3. **注册模型**：
* 在 `main.py` 的 `MODEL_DISPATCHER` 字典中添加映射：`"new_model": run_new_model.execute`。



## ⚠️ 注意事项

* **数据隐私**：`data/` 目录包含大量金融数据，已配置在 `.gitignore` 中，请勿强制提交到版本控制系统。
* **Redis 连接**：默认连接本地 Redis (`localhost:6379`)。如需修改，请调整 `models/run_rnn_common.py` 中的 `_push_to_redis` 函数。
* **GPU 显存**：Transformer 和 TRA 模型对显存要求较高，如遇 OOM (Out of Memory)，请在配置文件中调小 `batch_size`。

---

### 📝 TODO

* [ ] 集成 Qlib 的 Alpha158 因子库
* [ ] 增加模型融合 (Ensemble) 模块
* [ ] 对接实盘交易接口 (Trade Executor)

# Decider 目录结构说明

## 目录组织

```
decider/
├── __init__.py                 # 模块初始化
├── DIRECTORY_STRUCTURE.md      # 本文件
│
├── training/                   # 训练模块
│   ├── __init__.py
│   ├── grpo.py                 # GRPO 训练主脚本
│   ├── grpo_trainer.py         # GRPO Trainer 实现
│   ├── config.yaml             # 主配置文件
│   └── test_config.yaml        # 测试配置文件
│
├── data/                       # 数据准备模块
│   ├── __init__.py
│   ├── prepare_training_data.py    # 数据格式转换脚本
│   ├── test_input_data.json        # 测试输入数据
│   ├── example_input.json          # 示例输入数据
│   └── create_test_dataset.sh      # 创建测试数据集脚本
│
├── inference/                  # 推理模块
│   ├── __init__.py
│   ├── inference_api.py        # 推理 API
│   └── run_inference.py        # 推理运行脚本
│
├── version/                    # 版本管理模块
│   ├── __init__.py
│   └── version_manager.py      # 版本管理器
│
├── scripts/                    # 工具脚本
│   ├── __init__.py
│   ├── check_ready.py          # 训练前检查脚本
│   ├── quick_test.py           # 快速测试脚本
│   └── run_with_qwen2vl.sh     # Qwen2-VL 运行脚本
│
├── docs/                       # 文档
│   ├── CHECK_BEFORE_TRAINING.md
│   ├── CONFIG_EXPLANATION.md
│   ├── QWEN2VL_SETUP.md
│   ├── README_TEST.md
│   ├── README_data_preparation.md
│   └── VERSION_MANAGEMENT.md
│
├── configs/                    # 配置文件/资源
│   └── decision_prompt.txt     # 决策提示模板
│
└── outputs/                    # 输出目录
    └── test_output/            # 测试输出
```

## 模块说明

### training/ - 训练模块
包含所有训练相关的代码和配置：
- `grpo.py`: GRPO 训练主脚本，包含数据加载、训练流程
- `grpo_trainer.py`: GRPO Trainer 实现，支持 Qwen2-VL
- `config.yaml`: 主训练配置文件
- `test_config.yaml`: 测试用的小规模配置

**使用方式**:
```bash
cd decider
python training/grpo.py --config training/test_config.yaml
```

### data/ - 数据准备模块
包含数据准备和格式转换工具：
- `prepare_training_data.py`: 将原始数据转换为训练格式
- `test_input_data.json`: 测试输入数据示例
- `example_input.json`: 示例输入数据
- `create_test_dataset.sh`: 自动化数据准备脚本

**使用方式**:
```bash
cd decider
python data/prepare_training_data.py data/test_input_data.json --output-dir ./training_data
```

### inference/ - 推理模块
包含模型推理接口：
- `inference_api.py`: 推理 API，支持通过版本号加载模型
- `run_inference.py`: 简化的推理运行脚本

**使用方式**:
```bash
cd decider
python inference/run_inference.py \
    --version v20241212_001 \
    --report "数据集报告..." \
    --available-agents agent1 agent2
```

### version/ - 版本管理模块
包含版本管理功能：
- `version_manager.py`: 版本管理器，处理版本注册、查询、删除

**使用方式**:
```bash
cd decider
python version/version_manager.py list
python version/version_manager.py get --version v20241212_001
```

### scripts/ - 工具脚本
包含各种辅助工具：
- `check_ready.py`: 训练前检查脚本
- `quick_test.py`: 快速测试脚本
- `run_with_qwen2vl.sh`: Qwen2-VL 运行脚本

### docs/ - 文档
包含所有文档文件，说明各个模块的使用方法。

### configs/ - 配置文件
包含配置文件和资源文件：
- `decision_prompt.txt`: 决策提示模板

### outputs/ - 输出目录
训练和推理的输出目录。

## 路径更新说明

由于文件已重新组织，以下路径需要更新：

### 1. 训练脚本路径
```bash
# 之前
python grpo.py --config test_config.yaml

# 现在
python training/grpo.py --config training/test_config.yaml
```

### 2. 数据准备路径
```bash
# 之前
python prepare_training_data.py test_input_data.json

# 现在
python data/prepare_training_data.py data/test_input_data.json
```

### 3. 推理脚本路径
```bash
# 之前
python run_inference.py --version v20241212_001

# 现在
python inference/run_inference.py --version v20241212_001
```

### 4. 版本管理路径
```bash
# 之前
python version_manager.py list

# 现在
python version/version_manager.py list
```

## 导入路径更新

代码中的导入路径已自动更新：
- `from version_manager import ...` → `from version.version_manager import ...`
- `from inference_api import ...` → `from inference.inference_api import ...`

## 建议的工作流程

1. **准备数据**:
   ```bash
   cd decider
   python data/prepare_training_data.py data/test_input_data.json --output-dir ./training_data
   ```

2. **训练模型**:
   ```bash
   python training/grpo.py --config training/test_config.yaml
   ```

3. **查看版本**:
   ```bash
   python version/version_manager.py list
   ```

4. **运行推理**:
   ```bash
   python inference/run_inference.py --version v20241212_001 --report "..." --available-agents ...
   ```

## 注意事项

1. 运行脚本时，建议在 `decider/` 目录下执行，以确保相对路径正确
2. 配置文件中的路径可能需要更新（如 `dataset_name`）
3. 版本注册表 `model_registry.json` 会保存在 `decider/` 目录下
4. 模型版本会保存在 `decider/model_versions/` 目录下


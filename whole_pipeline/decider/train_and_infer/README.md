# Decider - 数据过滤决策模型

数据过滤决策模型的训练和推理系统，使用 GRPO 算法训练 Qwen2-VL 模型。

## 目录结构

```
decider/
├── training/          # 训练模块
├── data/              # 数据准备模块
├── inference/         # 推理模块
├── version/           # 版本管理模块
├── scripts/           # 工具脚本
├── docs/              # 文档
├── configs/           # 配置文件
└── outputs/           # 输出目录
```

详细目录结构说明请参考 [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

## 快速开始

### 1. 拉取并注册基础模型（首次使用）

从 HuggingFace 拉取模型并自动注册为初版：

```bash
cd decider
python scripts/download_and_register_model.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct
```

这会自动：
- 从 HuggingFace 下载模型
- 注册为版本 `v0_base`（或自动生成版本号）
- 保存到 `model_versions/` 目录

### 2. 使用未训练模型进行推理

拉取模型后，可以直接使用未训练的模型进行推理：

```bash
python inference/run_inference.py \
    --version v0_base \
    --report "数据集报告..." \
    --available-agents quality_filter_agent deduplication_filter
```

### 3. 准备训练数据

```bash
python data/prepare_training_data.py data/test_input_data.json --output-dir ./training_data
```

### 4. 训练模型

```bash
python training/grpo.py --config training/test_config.yaml
```

训练完成后会自动注册新版本（例如：`v20241212_001`）

### 5. 使用训练后的模型进行推理

```bash
python inference/run_inference.py \
    --version v20241212_001 \
    --report "数据集报告..." \
    --available-agents quality_filter_agent deduplication_filter
```

## 主要功能

- **模型拉取**: 从 HuggingFace 自动拉取模型并注册为初版
- **未训练模型推理**: 支持对未训练的基础模型进行推理
- **训练**: GRPO 算法训练，支持 Qwen2-VL
- **版本管理**: 自动版本注册和管理
- **推理接口**: 通过版本号调用模型（支持训练和未训练模型）
- **数据准备**: 数据格式转换工具

## 文档

- [目录结构说明](DIRECTORY_STRUCTURE.md)
- [配置说明](docs/CONFIG_EXPLANATION.md)
- [版本管理](docs/VERSION_MANAGEMENT.md)
- [数据准备](docs/README_data_preparation.md)
- [Qwen2-VL 设置](docs/QWEN2VL_SETUP.md)
- [测试指南](docs/README_TEST.md)

## 注意事项

- 运行脚本时，建议在 `decider/` 目录下执行
- 配置文件路径已更新，使用相对路径时需要从 `decider/` 目录运行
- 版本注册表保存在 `decider/model_registry.json`
- 模型版本保存在 `decider/model_versions/`


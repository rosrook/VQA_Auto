# 配置文件说明

本目录包含VQA Pipeline的所有配置文件。

## 文件说明

### `vqa_config.yaml`
完整的VQA任务配置文件，包含所有模块的配置选项。这是推荐使用的配置文件。

### `data_config.yaml`
数据模块的配置文件（保留用于向后兼容）。

### `config_template.yaml`
配置文件模板，可以复制此文件创建新的配置。

## 快速开始

### 1. 使用默认配置

```python
from pipeline import create_pipeline

pipeline = create_pipeline('config/vqa_config.yaml')
```

### 2. 创建自定义配置

1. 复制模板：
```bash
cp config/config_template.yaml config/my_config.yaml
```

2. 修改配置：
```yaml
# config/my_config.yaml
task_type: "vqa"
data_paths:
  train: "my_data/train.jsonl"
model:
  name: "Salesforce/blip-vqa-base"
training:
  num_epochs: 10
```

3. 使用配置：
```python
pipeline = create_pipeline('config/my_config.yaml')
```

## 配置项说明

### 基础配置
- `task_type`: 任务类型（vqa, image_captioning等）
- `experiment_name`: 实验名称（null=自动生成）

### 数据配置
- `data_paths`: 数据文件路径
- `data_split`: 数据分割配置
- `optimization`: 内存优化配置
- `image`: 图像处理配置
- `vqa`: VQA任务特定配置

### 模型配置
- `model`: 模型配置（名称、类型、任务等）
- `tokenizer`: Tokenizer配置
- `image_processor`: 图像处理器配置

### 训练配置
- `training`: 训练相关配置
  - `optimizer`: 优化器配置
  - `scheduler`: 学习率调度器配置
  - `early_stopping`: 早停配置
  - `freeze`: 冻结层配置

### 其他配置
- `evaluation`: 评估配置
- `preprocessing`: 预处理配置
- `logging`: 日志配置
- `inference`: 推理配置

## 配置示例

### 小数据集配置
```yaml
optimization:
  strategy: "standard"
  cache_images: true
  preload_images: true
dataloader:
  batch_size: 32
training:
  gradient_accumulation_steps: 1
  fp16: false
```

### 大数据集配置
```yaml
optimization:
  strategy: "streaming"
  use_streaming: true
dataloader:
  batch_size: 8
training:
  gradient_accumulation_steps: 8
  fp16: true
```

### BLIP模型配置
```yaml
model:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
```

## 注意事项

1. **路径配置**: 所有路径可以是相对路径或绝对路径
2. **null值**: 使用`null`表示不使用该功能
3. **自动检测**: 如果`type`字段为`null`或`auto`，系统会自动检测
4. **设备配置**: `device: null`表示自动选择设备（优先CUDA）

## 更多信息

详细配置说明请参考：
- `src/PIPELINE_GUIDE.md` - Pipeline使用指南
- `src/data/CREATE_DATALOADER_GUIDE.md` - 数据加载指南
- `src/models/MODEL_USAGE_GUIDE.md` - 模型使用指南
- `src/training/TRAINING_GUIDE.md` - 训练指南


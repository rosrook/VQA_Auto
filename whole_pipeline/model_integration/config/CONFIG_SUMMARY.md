# 配置文件总结

本目录包含VQA Pipeline的所有配置文件。

## 文件列表

### 1. `vqa_config.yaml` ⭐ **推荐使用**
完整的VQA任务配置文件，包含所有模块的配置选项：
- ✅ 数据配置（data_paths, data_split, optimization等）
- ✅ 模型配置（model, tokenizer, image_processor）
- ✅ 训练配置（training, optimizer, scheduler, early_stopping等）
- ✅ 评估配置（evaluation）
- ✅ 日志配置（logging）
- ✅ 推理配置（inference）

**使用方式：**
```python
from pipeline import create_pipeline
pipeline = create_pipeline('config/vqa_config.yaml')
```

### 2. `data_config.yaml`
数据模块的配置文件（保留用于向后兼容）。
- 包含数据加载、预处理、优化等配置
- 可以与新的pipeline一起使用

### 3. `config_template.yaml`
配置文件模板，用于快速创建新配置。
- 包含最常用的配置项
- 可以复制并修改

### 4. `README.md`
配置文件使用说明文档。

## 配置结构

### 基础配置
```yaml
task_type: "vqa"                    # 任务类型
experiment_name: null               # 实验名称（null=自动生成）
```

### 数据配置
```yaml
data_paths:
  train: "data/train.jsonl"
  validation: null
  test: null

data_split:
  auto_split: false
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

optimization:
  strategy: "auto"
  cache_images: true
  # ... 更多选项

image:
  root_dir: "data/images"
  # ... 更多选项

vqa:
  data_fields:
    image_field: "image"
    question_field: "question"
    answer_field: "answer"
```

### 模型配置
```yaml
model:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  task: "vqa"
  device: null                      # null=自动选择

tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  max_length: 128

image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  size: 384
```

### 训练配置
```yaml
training:
  num_epochs: 5
  fp16: true
  gradient_accumulation_steps: 2
  
  optimizer:
    type: "adamw"
    lr: 3e-5
    weight_decay: 0.01
  
  scheduler:
    type: "cosine"
  
  early_stopping:
    enabled: true
    monitor: "val_loss"
    patience: 3
  
  freeze:
    enabled: true
    layers: ["vision_model"]
```

### 其他配置
```yaml
evaluation:
  metrics: ["loss", "accuracy", "exact_match", "f1"]

preprocessing:
  batch_preprocess: true
  cache_processed: true

logging:
  level: "INFO"
  log_dir: "logs"

inference:
  max_length: 20
  num_beams: 3
```

## 快速开始

### 1. 使用默认配置
```python
from pipeline import create_pipeline

pipeline = create_pipeline('config/vqa_config.yaml')
pipeline.setup_data()
pipeline.setup_model()
pipeline.setup_training()
pipeline.train()
```

### 2. 创建自定义配置
```bash
# 复制模板
cp config/config_template.yaml config/my_config.yaml

# 编辑配置
vim config/my_config.yaml

# 使用配置
pipeline = create_pipeline('config/my_config.yaml')
```

### 3. 修改现有配置
直接编辑 `vqa_config.yaml` 文件，修改需要的配置项。

## 配置优先级

1. **命令行参数** > **配置文件** > **默认值**
2. Pipeline初始化时的参数会覆盖配置文件中的值

## 常见配置场景

### 场景1：小数据集训练
```yaml
optimization:
  strategy: "standard"
  preload_images: true
dataloader:
  batch_size: 32
training:
  gradient_accumulation_steps: 1
  fp16: false
```

### 场景2：大数据集训练
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

### 场景3：只训练分类头
```yaml
training:
  freeze:
    enabled: true
    layers: ["vision_model", "text_encoder"]
```

### 场景4：使用CPU训练
```yaml
model:
  device: "cpu"
training:
  fp16: false
```

### 场景5：量化训练
```yaml
model:
  load_in_8bit: true  # 或 load_in_4bit: true
```

## 配置验证

配置文件会在Pipeline初始化时自动验证：
- 必需的字段是否存在
- 字段类型是否正确
- 值是否在有效范围内

如果配置有误，会抛出相应的错误信息。

## 注意事项

1. **路径配置**：支持相对路径和绝对路径
2. **null值**：使用`null`表示不使用该功能或使用默认值
3. **自动检测**：`type: null`或`type: auto`会触发自动检测
4. **设备选择**：`device: null`会自动选择最佳设备（优先CUDA）
5. **YAML格式**：注意缩进和冒号后的空格

## 更多信息

- 详细配置说明：`README.md`
- Pipeline使用指南：`../src/PIPELINE_GUIDE.md`
- 数据加载指南：`../src/data/CREATE_DATALOADER_GUIDE.md`
- 模型使用指南：`../src/models/MODEL_USAGE_GUIDE.md`
- 训练指南：`../src/training/TRAINING_GUIDE.md`


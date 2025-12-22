# Pipeline使用指南

本指南展示如何使用主Pipeline进行完整的训练和推理流程。

## 快速开始

### 最简单的使用方式

```python
from pipeline import create_pipeline

# 创建管线
pipeline = create_pipeline('config/vqa_config.yaml', experiment_name='my_experiment')

# 设置并训练
pipeline.setup_data()
pipeline.setup_model()
pipeline.setup_training()
pipeline.train()

# 评估
results = pipeline.evaluate(split='validation')
print(f"验证准确率: {results['accuracy']:.4f}")

# 预测
answer = pipeline.predict('path/to/image.jpg', 'What color is the car?')
print(f"答案: {answer}")

# 保存
pipeline.save()
```

---

## 完整流程

### 1. 创建Pipeline

```python
from pipeline import create_pipeline

pipeline = create_pipeline(
    config_path='config/vqa_config.yaml',
    experiment_name='blip_vqa_experiment_001',
    log_dir='logs',
    log_level='INFO'
)
```

### 2. 设置数据

```python
# 自动从配置文件加载数据
pipeline.setup_data()

# 访问数据加载器
train_loader = pipeline.data_pipeline.get_train_dataloader()
val_loader = pipeline.data_pipeline.get_val_dataloader()
```

### 3. 设置模型

```python
# 使用配置文件中的模型设置
pipeline.setup_model()

# 或自定义模型
pipeline.setup_model(
    model_name='Salesforce/blip-vqa-base',
    model_type='blip',
    task='vqa',
    device='cuda',
    load_in_8bit=False  # 可选：8bit量化
)
```

### 4. 设置训练

```python
# 使用配置文件中的训练设置
pipeline.setup_training()

# 训练配置会自动从config文件中读取：
# - optimizer配置
# - scheduler配置
# - early_stopping配置
# - checkpoint配置
# - freeze配置
```

### 5. 开始训练

```python
pipeline.train()

# 训练过程会自动：
# - 记录日志
# - 保存检查点
# - 记录训练历史
# - 支持早停
# - TensorBoard日志（如果启用）
```

### 6. 评估模型

```python
# 评估验证集
val_results = pipeline.evaluate(split='validation')
print(f"验证损失: {val_results['loss']:.4f}")
print(f"验证准确率: {val_results['accuracy']:.4f}")

# 评估测试集
test_results = pipeline.evaluate(split='test', return_predictions=True)
predictions = test_results['predictions']
labels = test_results['labels']
```

### 7. 预测单个样本

```python
answer = pipeline.predict(
    image_path='path/to/image.jpg',
    question='What color is the car?',
    max_length=20,
    num_beams=3
)
print(f"答案: {answer}")
```

### 8. 保存模型

```python
# 保存到默认位置（checkpoints/{experiment_name}）
pipeline.save()

# 或指定保存路径
pipeline.save(save_dir='my_models/final_model')
```

---

## 配置文件示例

### 完整的配置文件结构

```yaml
# config/vqa_config.yaml

# 任务类型
task_type: "vqa"

# 数据路径
data_paths:
  train: "data/train.jsonl"
  validation: "data/val.jsonl"
  test: null

data_split:
  auto_split: false

# VQA任务配置
vqa:
  data_fields:
    image_field: "image"
    question_field: "question"
    answer_field: "answer"
  max_question_length: 128
  max_answer_length: 32

# Tokenizer配置
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  max_length: 128

# 图像Processor配置
image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"

# 图像配置
image:
  root_dir: "data/images"

# DataLoader配置
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 4
  pin_memory: true

# 模型配置
model:
  name: "Salesforce/blip-vqa-base"
  type: "blip"

# 训练配置
training:
  num_epochs: 5
  fp16: true
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  checkpoint_dir: "checkpoints"
  max_checkpoints: 5
  save_best_only: true
  use_tensorboard: true
  
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
    mode: "min"
  
  freeze:
    enabled: true
    layers: ["vision_model"]

# 日志配置
logging:
  level: "INFO"
  log_statistics: true
```

---

## 高级用法

### 1. 自定义训练配置

```python
pipeline = create_pipeline('config/vqa_config.yaml')

# 设置数据
pipeline.setup_data()

# 设置模型
pipeline.setup_model()

# 自定义训练设置
pipeline.setup_training(
    num_epochs=10,
    fp16=True,
    gradient_accumulation_steps=4
)

# 训练
pipeline.train()
```

### 2. 加载检查点继续训练

```python
pipeline = create_pipeline('config/vqa_config.yaml')
pipeline.setup_data()
pipeline.setup_model()
pipeline.setup_training()

# 加载检查点
checkpoint = pipeline.load_checkpoint('checkpoints/experiment_001/best_checkpoint.pt')

# 继续训练
pipeline.train()
```

### 3. 只进行评估（不训练）

```python
pipeline = create_pipeline('config/vqa_config.yaml')
pipeline.setup_data()
pipeline.setup_model()

# 加载训练好的模型
pipeline.load_checkpoint('checkpoints/best_model.pt')

# 评估
results = pipeline.evaluate(split='test')
print(results)
```

### 4. 批量预测

```python
pipeline = create_pipeline('config/vqa_config.yaml')
pipeline.setup_model()
pipeline.load_checkpoint('checkpoints/best_model.pt')

# 批量预测
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
questions = ['Question 1', 'Question 2', 'Question 3']

answers = []
for img, q in zip(images, questions):
    answer = pipeline.predict(img, q)
    answers.append(answer)

print(answers)
```

### 5. 访问训练历史

```python
pipeline.train()

# 访问训练历史
history = pipeline.training_logger.history
for epoch_data in history:
    print(f"Epoch {epoch_data['epoch']}:")
    print(f"  Train Loss: {epoch_data['train']['loss']:.4f}")
    print(f"  Val Loss: {epoch_data['val']['loss']:.4f}")

# 获取最佳epoch
best_epoch = pipeline.training_logger.get_best_epoch(metric='val_loss', mode='min')
print(f"最佳epoch: {best_epoch}")
```

---

## 工具模块使用

### 日志工具

```python
from utils import setup_logger, TrainingLogger

# 设置全局日志
logger = setup_logger(name='MyExperiment', log_dir='logs', level='INFO')
logger.info("训练开始")
logger.log_metrics({'loss': 0.5, 'accuracy': 0.8})

# 训练日志
training_logger = TrainingLogger(log_dir='logs/training', experiment_name='exp1')
training_logger.log_epoch(
    epoch=1,
    train_metrics={'loss': 0.5},
    val_metrics={'loss': 0.4}
)
```

### 检查点管理

```python
from utils import CheckpointManager

# 创建检查点管理器
manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    max_checkpoints=5,
    best_metric='val_loss',
    mode='min'
)

# 保存检查点
checkpoint_path = manager.save(
    model=model,
    optimizer=optimizer,
    epoch=1,
    metrics={'val_loss': 0.4},
    is_best=True
)

# 加载最佳检查点
checkpoint = manager.load(load_best=True)
model = manager.load_model(model, load_best=True)
```

### 评估指标

```python
from utils import compute_metrics, accuracy, exact_match, f1_score

predictions = ['red', 'blue', 'green']
labels = ['red', 'blue', 'yellow']

# 计算单个指标
acc = accuracy(predictions, labels)
em = exact_match(predictions, labels)
f1 = f1_score(predictions, labels, average='macro')

# 计算多个指标
metrics = compute_metrics(
    predictions,
    labels,
    metric_names=['accuracy', 'exact_match', 'f1', 'bleu', 'rouge']
)
print(metrics)
```

---

## 完整示例

### 示例1：完整训练流程

```python
from pipeline import create_pipeline

# 1. 创建管线
pipeline = create_pipeline(
    config_path='config/vqa_config.yaml',
    experiment_name='blip_vqa_finetune',
    log_dir='logs'
)

# 2. 设置数据
print("设置数据...")
pipeline.setup_data()

# 3. 设置模型
print("设置模型...")
pipeline.setup_model()

# 4. 设置训练
print("设置训练...")
pipeline.setup_training()

# 5. 训练
print("开始训练...")
pipeline.train()

# 6. 评估
print("评估模型...")
val_results = pipeline.evaluate(split='validation')
print(f"验证结果: {val_results}")

# 7. 保存
print("保存模型...")
pipeline.save()

print("完成！")
```

### 示例2：推理流程

```python
from pipeline import create_pipeline

# 1. 创建管线
pipeline = create_pipeline('config/vqa_config.yaml')

# 2. 设置模型
pipeline.setup_model()

# 3. 加载训练好的模型
pipeline.load_checkpoint('checkpoints/best_model.pt')

# 4. 预测
answer = pipeline.predict('test_image.jpg', 'What is in the image?')
print(f"答案: {answer}")
```

---

## 目录结构

训练完成后，会生成以下目录结构：

```
logs/
  └── {experiment_name}/
      ├── {experiment_name}.log              # 主日志文件
      ├── {experiment_name}_config.json      # 配置备份
      └── {experiment_name}_history.json     # 训练历史

checkpoints/
  └── {experiment_name}/
      ├── checkpoint_epoch_001_*.pt          # 检查点文件
      ├── best_checkpoint_epoch_*.pt        # 最佳检查点
      ├── checkpoint_index.json              # 检查点索引
      ├── training_log.csv                   # CSV日志
      └── tensorboard/                       # TensorBoard日志
```

---

## 常见问题

### Q1: 如何修改训练配置？

修改配置文件 `config/vqa_config.yaml`，然后重新创建pipeline。

### Q2: 如何恢复训练？

```python
pipeline = create_pipeline('config/vqa_config.yaml')
pipeline.setup_data()
pipeline.setup_model()
pipeline.setup_training()

# 加载检查点
pipeline.load_checkpoint('checkpoints/experiment/best_checkpoint.pt')

# 继续训练
pipeline.train()
```

### Q3: 如何使用多个GPU？

在配置文件中设置设备映射，或使用 `device_map='auto'`：

```python
pipeline.setup_model(device_map='auto')
```

### Q4: 如何监控训练过程？

启用TensorBoard：

```yaml
training:
  use_tensorboard: true
```

然后运行：
```bash
tensorboard --logdir checkpoints/{experiment_name}/tensorboard
```

---

## 总结

Pipeline模块提供了：

1. **统一的接口**：整合data、models、training、utils模块
2. **完整的流程**：从数据加载到模型训练、评估、推理
3. **自动化管理**：自动日志记录、检查点保存、训练历史
4. **易于使用**：简单的API，支持配置文件

使用Pipeline可以轻松完成整个VQA任务的训练和推理流程！


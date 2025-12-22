# 训练模块使用指南

本指南展示如何使用训练模块进行模型训练，集成data和model模块。

## 快速开始

### 方法1：使用配置文件（推荐）

```python
from training import create_trainer_from_config

# 创建训练器
trainer = create_trainer_from_config(
    data_config_path='config/vqa_config.yaml',
    model_name='Salesforce/blip-vqa-base',
    model_type='blip',
    task='vqa',
    num_epochs=3,
    fp16=True,
    save_dir='checkpoints/blip_vqa'
)

# 开始训练
trainer.train()
```

### 方法2：手动创建训练器

```python
from data.data_pipeline import DataPipeline
from models.model_loader import load_model
from training import Trainer, EarlyStoppingCallback, ModelCheckpointCallback

# 1. 加载数据
pipeline = DataPipeline('config/vqa_config.yaml')
pipeline.setup()
train_loader = pipeline.get_train_dataloader()
val_loader = pipeline.get_val_dataloader()

# 2. 加载模型
model_result = load_model(
    model_name='Salesforce/blip-vqa-base',
    model_type='blip',
    task='vqa',
    device='cuda'
)
model = model_result['model']

# 3. 创建优化器
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=3e-5)

# 4. 创建回调函数
callbacks = [
    EarlyStoppingCallback(monitor='val_loss', patience=5),
    ModelCheckpointCallback(save_dir='checkpoints', monitor='val_loss')
]

# 5. 创建训练器
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    callbacks=callbacks,
    num_epochs=3,
    fp16=True
)

# 6. 开始训练
trainer.train()
```

---

## 回调函数

### 早停回调

```python
from training import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    monitor='val_loss',      # 监控的指标
    patience=5,              # 容忍多少个epoch没有改善
    min_delta=0.001,         # 最小改善幅度
    mode='min',              # 'min' 或 'max'
    restore_best_weights=True  # 是否恢复最佳权重
)
```

### 模型检查点回调

```python
from training import ModelCheckpointCallback

checkpoint = ModelCheckpointCallback(
    save_dir='checkpoints',
    monitor='val_loss',
    mode='min',
    save_best_only=True,     # 是否只保存最佳模型
    save_frequency=1,        # 保存频率（每N个epoch）
    save_top_k=3,            # 保存top-k个最佳模型
    filename='checkpoint-{epoch:03d}-{val_loss:.4f}.pt'
)
```

### TensorBoard回调

```python
from training import TensorBoardCallback

tensorboard = TensorBoardCallback(log_dir='logs/tensorboard')
```

### CSV日志回调

```python
from training import CSVLoggerCallback

csv_logger = CSVLoggerCallback(filename='training_log.csv')
```

### 学习率调度器回调

```python
from training import LearningRateSchedulerCallback
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=10)
lr_callback = LearningRateSchedulerCallback(scheduler)
```

### 进度条回调

```python
from training import ProgressBarCallback

progress_bar = ProgressBarCallback(verbose=1)  # 0=静默, 1=进度条, 2=详细信息
```

---

## 评估器

### 通用评估器

```python
from training import Evaluator

evaluator = Evaluator(
    model=model,
    device='cuda',
    metrics=['loss', 'accuracy', 'f1']
)

# 评估
results = evaluator.evaluate(val_loader)
print(f"验证损失: {results['loss']:.4f}")
print(f"验证准确率: {results['accuracy']:.4f}")
```

### VQA专用评估器

```python
from training import VQAEvaluator

vqa_evaluator = VQAEvaluator(
    model=model,
    processor=processor,
    device='cuda'
)

# 评估
results = vqa_evaluator.evaluate(val_loader, return_predictions=True)
print(f"验证损失: {results['val_loss']:.4f}")
print(f"验证准确率: {results['accuracy']:.4f}")
print(f"完全匹配率: {results['exact_match']:.4f}")

# 获取预测结果
predictions = results['predictions']
labels = results['labels']
```

---

## 完整训练示例

### 示例1：BLIP VQA训练

```python
import logging
from training import create_trainer_from_config

logging.basicConfig(level=logging.INFO)

# 创建训练器
trainer = create_trainer_from_config(
    data_config_path='config/vqa_config.yaml',
    model_name='Salesforce/blip-vqa-base',
    model_type='blip',
    task='vqa',
    num_epochs=5,
    fp16=True,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    save_dir='checkpoints/blip_vqa',
    optimizer={
        'type': 'adamw',
        'lr': 3e-5,
        'weight_decay': 0.01
    },
    scheduler={
        'type': 'cosine'
    },
    early_stopping={
        'enabled': True,
        'monitor': 'val_loss',
        'patience': 3
    },
    use_tensorboard=True,
    freeze={
        'enabled': True,
        'layers': ['vision_model']  # 冻结视觉编码器
    }
)

# 开始训练
trainer.train()

# 训练历史
print("训练历史:")
for epoch_logs in trainer.history:
    print(f"Epoch {epoch_logs['epoch']}: "
          f"train_loss={epoch_logs['train_loss']:.4f}, "
          f"val_loss={epoch_logs.get('val_loss', 'N/A')}")
```

### 示例2：自定义训练循环

```python
from data.data_pipeline import DataPipeline
from models.model_loader import load_model
from training import Trainer, VQAEvaluator
from training.callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ProgressBarCallback
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. 加载数据
pipeline = DataPipeline('config/vqa_config.yaml')
pipeline.setup()
train_loader = pipeline.get_train_dataloader()
val_loader = pipeline.get_val_dataloader()

# 2. 加载模型
model_result = load_model(
    model_name='Salesforce/blip-vqa-base',
    model_type='blip',
    task='vqa',
    device='cuda',
    load_processor=True
)
model = model_result['model']
processor = model_result['processor']

# 3. 创建优化器和调度器
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=5)

# 4. 创建评估器
evaluator = VQAEvaluator(model, processor, device='cuda')

# 5. 创建回调函数
callbacks = [
    ProgressBarCallback(verbose=1),
    EarlyStoppingCallback(monitor='val_loss', patience=3),
    ModelCheckpointCallback(
        save_dir='checkpoints',
        monitor='val_loss',
        save_best_only=True
    )
]

# 6. 创建训练器
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=callbacks,
    evaluator=evaluator,
    num_epochs=5,
    fp16=True,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0
)

# 7. 训练
trainer.train()
```

---

## 训练配置示例

### 配置文件格式

```yaml
# data_config.yaml (数据配置)
task_type: "vqa"
data_paths:
  train: "data/train.jsonl"
  validation: "data/val.jsonl"
  test: null

# training_config.yaml (训练配置，可选)
training:
  num_epochs: 5
  fp16: true
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  save_dir: "checkpoints"
  
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

---

## 高级功能

### 1. 梯度累积

```python
trainer = Trainer(
    ...,
    gradient_accumulation_steps=4  # 累积4个batch的梯度
)
```

### 2. 混合精度训练

```python
trainer = Trainer(
    ...,
    fp16=True  # 启用FP16混合精度训练
)
```

### 3. 梯度裁剪

```python
trainer = Trainer(
    ...,
    max_grad_norm=1.0  # 梯度裁剪阈值
)
```

### 4. 冻结层

```python
from models.model_utils import freeze_model

# 冻结视觉编码器
freeze_model(model, freeze_layers=['vision_model'])

# 或通过配置
trainer = create_trainer_from_config(
    ...,
    freeze={
        'enabled': True,
        'layers': ['vision_model']
    }
)
```

### 5. 保存和加载检查点

```python
# 保存检查点
trainer.save_checkpoint('checkpoints/epoch_5.pt')

# 加载检查点
trainer.load_checkpoint('checkpoints/epoch_5.pt')
```

---

## 常见问题

### Q1: 如何监控训练过程？

使用TensorBoard：

```python
from training import TensorBoardCallback

callbacks = [
    TensorBoardCallback(log_dir='logs/tensorboard'),
    ...
]

# 然后运行: tensorboard --logdir logs/tensorboard
```

### Q2: 如何实现自定义回调？

```python
from training.callbacks import Callback

class MyCallback(Callback):
    def on_batch_end(self, trainer, batch, logs, **kwargs):
        # 自定义逻辑
        if batch % 100 == 0:
            print(f"Batch {batch}: {logs}")

callbacks = [MyCallback(), ...]
```

### Q3: 如何在不同设备上训练？

```python
# CPU
trainer = create_trainer_from_config(..., device='cpu')

# 特定GPU
trainer = create_trainer_from_config(..., device='cuda:0')

# 自动选择
trainer = create_trainer_from_config(..., device=None)
```

### Q4: 如何处理内存不足？

1. 减小batch_size
2. 使用梯度累积
3. 使用混合精度训练
4. 冻结部分层

```python
trainer = create_trainer_from_config(
    ...,
    fp16=True,
    gradient_accumulation_steps=4,
    freeze={'enabled': True, 'layers': ['vision_model']}
)
```

---

## 总结

训练模块提供了：

1. **Trainer**: 完整的训练循环，支持梯度累积、混合精度等
2. **Callbacks**: 丰富的回调函数，包括早停、检查点、TensorBoard等
3. **Evaluator**: 模型评估功能，支持VQA等任务

与data和model模块完美集成，可以轻松构建完整的训练流程。


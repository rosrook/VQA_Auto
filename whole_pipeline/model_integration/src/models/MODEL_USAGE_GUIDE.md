# 模型加载和使用指南

本指南展示如何使用 `models` 模块加载和使用模型，特别是BLIP模型。

## 快速开始

### 方法1：使用便捷函数（推荐）

```python
from models import load_model

# 加载BLIP VQA模型
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    device="cuda",  # 或 "cpu"
    load_processor=True
)

model = result['model']
processor = result['processor']
tokenizer = result['tokenizer']
image_processor = result['image_processor']
```

### 方法2：使用ModelLoader类

```python
from models import ModelLoader

# 创建加载器
loader = ModelLoader(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    device="cuda"
)

# 加载模型和processor
result = loader.load(load_processor=True)

# 获取模型和processor
model = loader.get_model()
processor = loader.get_processor()
tokenizer = loader.get_tokenizer()
image_processor = loader.get_image_processor()
```

---

## 与DataPipeline集成

### 完整示例：加载数据并训练

```python
from data_pipeline import DataPipeline
from models import load_model, get_model_info, print_model_summary

# 1. 加载数据
pipeline = DataPipeline('config/vqa_config.yaml')
pipeline.setup()
train_loader = pipeline.get_train_dataloader()

# 2. 加载模型
model_result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    device="cuda",
    load_processor=True
)
model = model_result['model']

# 3. 查看模型信息
print_model_summary(model)

# 4. 训练循环
for batch in train_loader:
    pixel_values = batch['pixel_values'].to('cuda')
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    labels = batch['labels'].to('cuda')
    
    # 前向传播
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    # ... 反向传播和优化 ...
    break
```

---

## 模型工具函数

### 1. 统计参数量

```python
from models import count_parameters, get_model_size_mb

param_info = count_parameters(model)
print(f"总参数量: {param_info['total']:,}")
print(f"可训练参数: {param_info['trainable']:,}")
print(f"参数量（百万）: {param_info['total_millions']:.2f}M")

size_mb = get_model_size_mb(model)
print(f"模型大小: {size_mb:.2f} MB")
```

### 2. 冻结/解冻层

```python
from models import freeze_model, unfreeze_model

# 冻结所有层
freeze_model(model)

# 冻结特定层（例如只训练分类头）
freeze_model(model, freeze_layers=['vision_model', 'text_encoder'])

# 解冻所有层
unfreeze_model(model)

# 解冻特定层
unfreeze_model(model, unfreeze_layers=['classifier'])
```

### 3. 设备管理

```python
from models import get_device, move_model_to_device

# 获取模型所在设备
device = get_device(model)
print(f"模型在设备: {device}")

# 移动模型到指定设备
move_model_to_device(model, "cuda:0")
```

### 4. 保存和加载模型

```python
from models import save_model, load_model_from_path

# 保存模型
save_model(
    model=model,
    save_path="checkpoints/my_model",
    tokenizer=tokenizer,
    processor=processor
)

# 加载模型
from transformers import BlipForQuestionAnswering
loaded_model = load_model_from_path(
    model_path="checkpoints/my_model",
    model_class=BlipForQuestionAnswering,
    device="cuda"
)
```

### 5. 使用BLIP生成答案

```python
from models import generate_answer_blip
from PIL import Image

# 加载图像和问题
image = Image.open("path/to/image.jpg")
question = "What color is the car?"

# 生成答案
answer = generate_answer_blip(
    model=model,
    processor=processor,
    image=image,
    question=question,
    max_length=20,
    num_beams=3
)

print(f"答案: {answer}")
```

### 6. 准备输入

```python
from models import prepare_inputs_for_blip

# 准备单个输入
inputs = prepare_inputs_for_blip(
    processor=processor,
    images=image,
    texts=question,
    device="cuda"
)

# 准备批量输入
images = [image1, image2, image3]
questions = ["Question 1", "Question 2", "Question 3"]
inputs = prepare_inputs_for_blip(
    processor=processor,
    images=images,
    texts=questions,
    device="cuda"
)
```

### 7. 模型信息

```python
from models import get_model_info, print_model_summary

# 获取模型信息字典
info = get_model_info(model)
print(f"模型类型: {info['model_class']}")
print(f"参数量: {info['total_parameters_millions']:.2f}M")
print(f"设备: {info['device']}")

# 打印模型摘要
print_model_summary(model, detailed=True)
```

---

## 支持的模型类型

### BLIP

```python
# VQA任务
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa"
)

# Image Captioning任务
result = load_model(
    model_name="Salesforce/blip-image-captioning-base",
    model_type="blip",
    task="conditional_generation"
)

# Image-Text Retrieval任务
result = load_model(
    model_name="Salesforce/blip-base",
    model_type="blip",
    task="image_text_retrieval"
)
```

### Auto（自动检测）

```python
# 不指定model_type，自动检测
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    task="vqa"
)
```

---

## 高级用法

### 量化加载

```python
# 8bit量化
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    load_in_8bit=True
)

# 4bit量化（需要bitsandbytes）
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    load_in_4bit=True
)
```

### 指定数据类型

```python
import torch

result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    torch_dtype=torch.float16  # 或 torch.bfloat16
)
```

### 多GPU支持

```python
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    device_map="auto"  # 自动分配到多个GPU
)
```

---

## 完整训练示例

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from data_pipeline import DataPipeline
from models import load_model, freeze_model, print_model_summary

# 1. 加载数据
pipeline = DataPipeline('config/vqa_config.yaml')
pipeline.setup()
train_loader = pipeline.get_train_dataloader()

# 2. 加载模型
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    model_type="blip",
    task="vqa",
    device="cuda",
    load_processor=True
)
model = result['model']

# 3. 冻结视觉编码器（可选）
freeze_model(model, freeze_layers=['vision_model'])

# 4. 打印模型信息
print_model_summary(model)

# 5. 设置优化器
optimizer = AdamW(model.parameters(), lr=3e-5)

# 6. 训练循环
model.train()
for epoch in range(3):
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # 准备输入
        pixel_values = batch['pixel_values'].to('cuda')
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        
        # 前向传播
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch} 平均损失: {total_loss / len(train_loader):.4f}")

# 7. 保存模型
from models import save_model
save_model(
    model=model,
    save_path="checkpoints/blip_vqa_finetuned",
    tokenizer=result['tokenizer'],
    processor=result['processor']
)
```

---

## 常见问题

### Q1: 如何只加载模型而不加载processor？

```python
result = load_model(
    model_name="Salesforce/blip-vqa-base",
    load_processor=False  # 不加载processor
)
model = result['model']
# result中没有processor、tokenizer等
```

### Q2: 如何在不同设备上加载模型？

```python
# CPU
result = load_model(..., device="cpu")

# 特定GPU
result = load_model(..., device="cuda:0")

# 自动选择
result = load_model(..., device=None)  # 自动选择CUDA或CPU
```

### Q3: 如何检查模型是否支持某个任务？

```python
from models import ModelLoader

loader = ModelLoader("Salesforce/blip-vqa-base", model_type="blip")
# 检查支持的任务
print(ModelLoader.MODEL_REGISTRY['blip'].keys())
```

### Q4: 模型加载失败怎么办？

1. 检查网络连接（需要下载模型）
2. 检查模型名称是否正确
3. 检查是否有足够的磁盘空间
4. 尝试使用 `trust_remote_code=True`

```python
result = load_model(
    model_name="...",
    trust_remote_code=True
)
```

---

## 总结

`models` 模块提供了：

1. **ModelLoader**: 统一的模型加载接口，支持BLIP等多种模型
2. **工具函数**: 参数量统计、层冻结、设备管理、保存加载等
3. **便捷函数**: 快速加载模型和使用BLIP生成答案

与 `data_pipeline` 完美集成，可以轻松构建完整的训练和推理流程。


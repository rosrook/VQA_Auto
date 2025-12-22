# 从JSONL文件创建DataLoader指南

本指南展示如何使用 `src/data` 文件夹中的代码，从包含 `{"image", "question", "answer"}` 字段的JSONL文件创建DataLoader。

## 方法1：使用DataPipeline（推荐，最简单）

### 步骤1：准备配置文件

创建配置文件 `config/vqa_config.yaml`：

```yaml
# 任务类型
task_type: "vqa"

# 数据路径
data_paths:
  train: "path/to/your/data.jsonl"  # 你的JSONL文件路径
  validation: null  # 不需要验证集
  test: null        # 不需要测试集

# 不进行数据分割
data_split:
  auto_split: false

# VQA任务配置
vqa:
  data_fields:
    image_field: "image"      # JSONL中的图像字段名
    question_field: "question"  # JSONL中的问题字段名
    answer_field: "answer"      # JSONL中的答案字段名
  max_question_length: 128
  max_answer_length: 32

# Tokenizer配置（BLIP示例）
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"  # 可选：明确指定类型
  max_length: 128
  padding: "max_length"
  truncation: true

# 图像Processor配置（BLIP示例）
image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"  # 可选：明确指定类型
  size: 384

# 图像配置
image:
  root_dir: "path/to/images"  # 如果image字段是相对路径，指定图像根目录
  # 如果image字段是绝对路径或URL，可以设为null

# DataLoader配置
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 4
  pin_memory: true
  drop_last: false

# 日志配置
logging:
  level: "INFO"
  log_statistics: true
```

### 步骤2：使用DataPipeline

```python
from data_pipeline import DataPipeline

# 初始化并设置管线
pipeline = DataPipeline('config/vqa_config.yaml')
pipeline.setup()

# 获取训练DataLoader
train_loader = pipeline.get_train_dataloader()

# 使用DataLoader
for batch in train_loader:
    print("Batch keys:", batch.keys())
    print("Image shape:", batch['pixel_values'].shape)
    print("Question shape:", batch['input_ids'].shape)
    print("Answer shape:", batch['labels'].shape)
    break
```

### 完整示例代码

```python
"""
从JSONL文件创建VQA DataLoader的完整示例
"""
import sys
from pathlib import Path

# 添加路径（如果需要）
sys.path.append(str(Path(__file__).parent))

from data_pipeline import DataPipeline

def main():
    # 配置文件路径
    config_path = "config/vqa_config.yaml"
    
    # 创建并设置数据管线
    print("初始化数据管线...")
    pipeline = DataPipeline(config_path)
    pipeline.setup()
    
    # 获取训练DataLoader
    print("\n获取训练DataLoader...")
    train_loader = pipeline.get_train_dataloader()
    
    print(f"DataLoader创建成功！")
    print(f"数据集大小: {len(pipeline.get_dataset('train'))}")
    
    # 测试一个batch
    print("\n测试第一个batch...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Keys: {batch.keys()}")
        print(f"  Image shape: {batch['pixel_values'].shape}")
        print(f"  Question input_ids shape: {batch['input_ids'].shape}")
        print(f"  Question attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Answer labels shape: {batch['labels'].shape}")
        
        if batch_idx >= 0:  # 只测试第一个batch
            break

if __name__ == "__main__":
    main()
```

---

## 方法2：手动步骤（更灵活）

如果你需要更多控制，可以手动使用各个组件：

### 步骤1：加载数据

```python
from data_loader import DataLoader
from data_processor import DataProcessor
from dataset import VQADataset, create_dataloader
from transformers import AutoTokenizer, AutoImageProcessor

# 1. 加载JSONL文件
data_path = "path/to/your/data.jsonl"
loader = DataLoader(data_path)
raw_data = loader.load()

print(f"加载了 {len(raw_data)} 条数据")
print(f"第一条数据字段: {raw_data[0].keys()}")
```

### 步骤2：预处理数据

```python
# 2. 预处理数据（清理、标准化等）
preprocessing_config = {
    'processor_type': 'blip',  # 可选
    'processor_name': 'Salesforce/blip-vqa-base'  # 可选
}
processor = DataProcessor(preprocessing_config)
processed_data = processor.process(raw_data, task_type='vqa')

print(f"预处理完成，共 {len(processed_data)} 条数据")
```

### 步骤3：加载Tokenizer和Image Processor

```python
# 3. 加载tokenizer和image processor
from transformers import AutoProcessor  # 官方推荐方式

model_name = "Salesforce/blip-vqa-base"

# 方式1：使用完整的processor（推荐）
# 使用AutoProcessor（官方推荐方式）
# AutoProcessor会自动选择正确的processor类（如BlipProcessor）
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
image_processor = processor.image_processor

# 方式2：分别加载
# 或者分别加载（如果需要）
# from transformers import AutoTokenizer, AutoImageProcessor
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# image_processor = AutoImageProcessor.from_pretrained(model_name)
```

### 步骤4：创建Dataset

```python
# 4. 创建VQA Dataset
vqa_dataset = VQADataset(
    data=processed_data,
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_length=128,
    image_field="image",
    question_field="question",
    answer_field="answer",
    image_root="path/to/images",  # 如果image字段是相对路径
    return_raw_image=False  # 是否返回原始图像
)

print(f"Dataset创建成功，样本数: {len(vqa_dataset)}")
```

### 步骤5：创建DataLoader

```python
# 5. 创建DataLoader
train_loader = create_dataloader(
    dataset=vqa_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print("DataLoader创建成功！")
```

### 步骤6：使用DataLoader

```python
# 6. 使用DataLoader
for batch_idx, batch in enumerate(train_loader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Image shape: {batch['pixel_values'].shape}")
    print(f"  Question input_ids shape: {batch['input_ids'].shape}")
    print(f"  Question attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Answer labels shape: {batch['labels'].shape}")
    
    # 你的训练代码...
    # model_output = model(pixel_values=batch['pixel_values'], 
    #                      input_ids=batch['input_ids'],
    #                      attention_mask=batch['attention_mask'],
    #                      labels=batch['labels'])
    
    if batch_idx >= 2:  # 只测试前3个batch
        break
```

### 完整手动示例代码

```python
"""
手动创建VQA DataLoader的完整示例
"""
from data_loader import DataLoader
from data_processor import DataProcessor
from dataset import VQADataset, create_dataloader
from transformers import AutoProcessor  # 官方推荐方式

def create_vqa_dataloader(
    jsonl_path: str,
    image_root: str = None,
    model_name: str = "Salesforce/blip-vqa-base",
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 4
):
    """
    从JSONL文件创建VQA DataLoader
    
    Args:
        jsonl_path: JSONL文件路径
        image_root: 图像根目录（如果image字段是相对路径）
        model_name: HuggingFace模型名称
        batch_size: 批次大小
        max_length: 最大文本长度
        num_workers: 数据加载进程数
        
    Returns:
        DataLoader对象
    """
    # 1. 加载数据
    print("加载数据...")
    loader = DataLoader(jsonl_path)
    raw_data = loader.load()
    print(f"  加载了 {len(raw_data)} 条数据")
    
    # 2. 预处理数据
    print("预处理数据...")
    processor = DataProcessor()
    processed_data = processor.process(raw_data, task_type='vqa')
    print(f"  预处理完成，共 {len(processed_data)} 条数据")
    
    # 3. 加载processor
    print(f"加载processor: {model_name}...")
    # 使用AutoProcessor（官方推荐方式）
    # AutoProcessor会自动选择正确的processor类（如BlipProcessor）
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    print("  Processor加载完成")
    
    # 4. 创建Dataset
    print("创建Dataset...")
    vqa_dataset = VQADataset(
        data=processed_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length,
        image_field="image",
        question_field="question",
        answer_field="answer",
        image_root=image_root,
        return_raw_image=False
    )
    print(f"  Dataset创建成功，样本数: {len(vqa_dataset)}")
    
    # 5. 创建DataLoader
    print("创建DataLoader...")
    dataloader = create_dataloader(
        dataset=vqa_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    print("  DataLoader创建成功！")
    
    return dataloader

# 使用示例
if __name__ == "__main__":
    # 创建DataLoader
    train_loader = create_vqa_dataloader(
        jsonl_path="data/vqa_train.jsonl",
        image_root="data/images",  # 如果image字段是相对路径
        model_name="Salesforce/blip-vqa-base",
        batch_size=16,
        max_length=128
    )
    
    # 测试
    print("\n测试DataLoader...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Image shape: {batch['pixel_values'].shape}")
        print(f"  Question shape: {batch['input_ids'].shape}")
        print(f"  Answer shape: {batch['labels'].shape}")
        
        if batch_idx >= 0:
            break
```

---

## JSONL文件格式要求

你的JSONL文件应该每行一个JSON对象，格式如下：

```jsonl
{"image": "path/to/image1.jpg", "question": "What color is the car?", "answer": "red"}
{"image": "path/to/image2.jpg", "question": "How many people are there?", "answer": "three"}
{"image": "path/to/image3.jpg", "question": "Is it sunny?", "answer": "yes"}
```

### 图像路径格式支持

`image` 字段可以是以下格式之一：

1. **相对路径**（需要设置 `image_root`）：
   ```json
   {"image": "images/001.jpg", ...}
   ```

2. **绝对路径**：
   ```json
   {"image": "/absolute/path/to/image.jpg", ...}
   ```

3. **URL**：
   ```json
   {"image": "https://example.com/image.jpg", ...}
   ```

4. **Base64编码**：
   ```json
   {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...", ...}
   ```

---

## 常见问题

### Q1: 如果我的JSONL文件字段名不同怎么办？

在配置文件中修改 `vqa.data_fields`：

```yaml
vqa:
  data_fields:
    image_field: "image_path"      # 你的图像字段名
    question_field: "q"            # 你的问题字段名
    answer_field: "ans"            # 你的答案字段名
```

### Q2: 如果图像路径是相对路径怎么办？

在配置文件中设置 `image.root_dir`：

```yaml
image:
  root_dir: "path/to/images"  # 图像根目录
```

### Q3: 如何使用其他模型（如CLIP）？

修改配置文件中的 `tokenizer` 和 `image_processor`：

```yaml
tokenizer:
  name: "openai/clip-vit-base-patch32"
  type: "clip"

image_processor:
  name: "openai/clip-vit-base-patch32"
  type: "clip"
```

### Q4: 如何只使用部分数据？

在加载数据后手动切片：

```python
# 方法1：使用DataPipeline
pipeline = DataPipeline(config_path)
pipeline.setup()
dataset = pipeline.get_dataset('train')

# 只使用前1000条数据
from torch.utils.data import Subset
subset = Subset(dataset, range(1000))
dataloader = create_dataloader(subset, batch_size=16)

# 方法2：手动加载时切片
raw_data = loader.load()
processed_data = processor.process(raw_data[:1000], task_type='vqa')
```

### Q5: 如何处理图像加载失败？

代码已经内置了错误处理：
- 如果图像加载失败，会自动创建一个空白图像作为fallback
- 错误会记录在日志中，但不会中断数据加载

---

## 总结

**推荐使用方法1（DataPipeline）**，因为：
- ✅ 最简单，只需配置文件
- ✅ 自动处理所有步骤
- ✅ 支持内存优化和自动检测
- ✅ 统一的接口

**使用方法2（手动步骤）**，当你需要：
- 更细粒度的控制
- 自定义处理逻辑
- 调试和测试


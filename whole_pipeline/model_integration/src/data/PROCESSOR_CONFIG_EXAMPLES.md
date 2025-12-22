# Processor配置示例

`data_pipeline.py` 现在支持多种processor类型（BLIP、CLIP等），不再局限于AutoProcessor。

## 配置方式

### 方式1：指定processor类型（推荐）

在配置文件中明确指定processor类型，系统会使用对应的processor类：

```yaml
# Tokenizer配置
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"  # 指定使用BLIP tokenizer
  max_length: 128
  padding: "max_length"
  truncation: true

# 图像Processor配置
image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"  # 指定使用BLIP processor
  size: 384
```

### 方式2：自动检测（默认）

如果不指定`type`，系统会自动检测processor类型：

```yaml
# Tokenizer配置
tokenizer:
  name: "Salesforce/blip-vqa-base"  # 系统会自动检测为BLIP
  max_length: 128

# 图像Processor配置
image_processor:
  name: "Salesforce/blip-vqa-base"  # 系统会自动检测为BLIP
  size: 384
```

### 方式3：使用Auto（回退）

如果自动检测失败，系统会自动回退到AutoProcessor/AutoTokenizer：

```yaml
tokenizer:
  name: "bert-base-uncased"
  type: "auto"  # 或省略type字段

image_processor:
  name: "google/vit-base-patch16-224"
  type: "auto"  # 或省略type字段
```

## 支持的Processor类型

### BLIP

```yaml
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"

image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"

# 支持的模型ID：
# - Salesforce/blip-vqa-base
# - Salesforce/blip-vqa-capfilt-large
# - Salesforce/blip-image-captioning-base
# - Salesforce/blip-image-captioning-large
```

### CLIP

```yaml
tokenizer:
  name: "openai/clip-vit-base-patch32"
  type: "clip"

image_processor:
  name: "openai/clip-vit-base-patch32"
  type: "clip"

# 支持的模型ID：
# - openai/clip-vit-base-patch32
# - openai/clip-vit-base-patch16
# - openai/clip-vit-large-patch14
```

### Auto（默认）

```yaml
tokenizer:
  name: "bert-base-uncased"
  type: "auto"  # 或省略

image_processor:
  name: "google/vit-base-patch16-224"
  type: "auto"  # 或省略
```

## 完整配置示例

### BLIP VQA任务

```yaml
task_type: "vqa"

# Tokenizer配置
tokenizer:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  max_length: 128
  padding: "max_length"
  truncation: true

# 图像Processor配置
image_processor:
  name: "Salesforce/blip-vqa-base"
  type: "blip"
  size: 384

# VQA任务配置
vqa:
  data_fields:
    image_field: "image"
    question_field: "question"
    answer_field: "answer"
  max_question_length: 128
  max_answer_length: 32
```

### CLIP任务

```yaml
task_type: "image_captioning"

tokenizer:
  name: "openai/clip-vit-base-patch32"
  type: "clip"
  max_length: 77

image_processor:
  name: "openai/clip-vit-base-patch32"
  type: "clip"
  size: 224
```

## 工作原理

1. **指定类型时**：
   - 系统从`DataProcessor.PROCESSOR_REGISTRY`中查找对应的processor类
   - 使用指定的类（如`BlipProcessor`、`BlipTokenizer`）加载模型
   - 如果加载失败，回退到AutoProcessor/AutoTokenizer

2. **自动检测时**：
   - 系统使用`DataProcessor`的自动检测功能
   - 根据模型名称中的关键词（如"blip"、"clip"）判断类型
   - 如果检测失败，使用AutoProcessor/AutoTokenizer

3. **回退机制**：
   - 所有加载方法都有多层回退机制
   - 确保即使指定类型失败，也能正常加载模型

## 优势

1. **更准确**：使用正确的processor类，避免AutoProcessor可能的问题
2. **更灵活**：支持多种processor类型，易于扩展
3. **向后兼容**：如果不指定类型，行为与之前一致
4. **自动回退**：即使指定类型失败，也能正常工作

## 注意事项

1. **类型匹配**：确保`tokenizer.type`和`image_processor.type`与模型匹配
2. **模型ID**：使用正确的HuggingFace模型ID
3. **回退机制**：如果遇到问题，可以尝试设置`type: "auto"`或省略`type`字段


# 模型调用代码执行流程详解

## 📋 整体流程概览

```
用户启动测试
    ↓
TestExecutor.run_benchmark()  [入口点]
    ↓
选择处理模式（流式/批量）
    ↓
_process_task_batch()  [核心处理函数]
    ↓
对每个任务：
    ├─ _build_prompt()  [构建prompt]
    ├─ model_adapter.generate()  [调用模型]
    ├─ 提取答案
    ├─ benchmark.evaluate_answer()  [评估答案]
    └─ 收集结果
```

---

## 🔍 详细流程解析

### 1️⃣ 入口点：`run_benchmark()`

**文件位置**: `core/test_executor.py` 第149行

**作用**: 测试执行的入口函数，决定使用流式还是批量处理模式

**关键代码**:
```python
def run_benchmark(self, benchmark, max_samples, batch_size, verbose):
    # 检查是否支持流式加载
    use_streaming = hasattr(benchmark, 'get_dataset_iterator') and 
                    hasattr(benchmark, '_use_streaming') and 
                    benchmark._use_streaming
    
    if use_streaming:
        # 流式处理模式
        results = self._run_benchmark_streaming(...)
    else:
        # 批量处理模式
        tasks = benchmark.get_tasks()
        results = self._process_tasks(benchmark, tasks_iter, batch_size)
```

**流程分支**:
- **流式模式**: 数据集很大时，逐项加载和处理
- **批量模式**: 数据集较小时，一次性加载所有任务

---

### 2️⃣ 任务处理：`_process_task_batch()`

**文件位置**: `core/test_executor.py` 第417行

**作用**: 处理一批任务，每个任务包含一个问题和图像

**执行步骤**:

#### 步骤1: 构建Prompt (第424行)
```python
prompt = self._build_prompt(task)
```

#### 步骤2: 调用模型 (第432-435行)
```python
model_response = self.model_adapter.generate(
    prompt=prompt,
    images=task.images
)
```

#### 步骤3: 提取答案 (第457行)
```python
model_answer = model_response.get("text", "")
```

#### 步骤4: 评估答案 (第460-464行)
```python
result = benchmark.evaluate_answer(
    model_answer=model_answer,
    ground_truth=task.ground_truth,
    task=task
)
```

---

### 3️⃣ Prompt构建：`_build_prompt()`

**文件位置**: `core/test_executor.py` 第494行

**作用**: 将任务中的问题转换为模型可理解的prompt格式

**当前实现**:
```python
def _build_prompt(self, task: BenchmarkTask) -> str:
    prompt = task.question  # 直接使用问题文本
    return prompt
```

**可扩展性**: 
- 可以根据不同模型类型调整prompt格式
- 例如：`f"Question: {task.question} Answer:"`

---

### 4️⃣ 模型调用：`model_adapter.generate()`

**文件位置**: `core/model_adapter.py` 第835行

**这是整个流程的核心！** 让我们详细看看：

#### 4.1 图像预处理 (第841-865行)

```python
# 处理不同类型的图像输入
pil_images = []
for img_path in images:
    if isinstance(img_path, str):
        if img_path.startswith("http"):
            # 从URL加载图像
        elif img_path.startswith("data:image"):
            # Base64编码的图像
        else:
            # 本地文件路径
            pil_images.append(Image.open(img_path))
    else:
        # 已经是PIL Image对象
        pil_images.append(img_path)
```

#### 4.2 模型生成（三种方法）

##### 方法1: 使用Processor（视觉-语言模型） (第868-914行)

**适用于**: BLIP、CLIP等使用processor的模型

```python
if self.has_processor and pil_images:
    # 1. 使用processor处理文本和图像
    inputs = self.processor(
        text=prompt,
        images=pil_images,
        return_tensors="pt"
    )
    
    # 2. 移动到正确的设备（GPU/CPU）
    inputs = {k: v.to(self.device) if isinstance(v, Tensor) else v 
             for k, v in inputs.items()}
    
    # 3. 保存输入长度（用于后续只解码新生成的部分）
    input_length = inputs['input_ids'].shape[1]
    
    # 4. 模型生成
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
    
    # 5. 只解码新生成的部分（关键步骤！）
    if input_length > 0 and len(outputs[0]) > input_length:
        generated_ids = outputs[0][input_length:]  # 只取新生成的部分
        generated_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )
```

**关键点**: 
- 使用 `outputs[0][input_length:]` 只解码新生成的token
- 避免重复包含输入prompt的问题

##### 方法2: 使用Chat接口 (第916-935行)

**适用于**: 支持chat接口的模型（如Qwen-VL）

```python
elif hasattr(self.model, 'chat') and pil_images:
    response, _ = self.model.chat(
        self.processor,
        query=prompt,
        history=None,
        images=pil_images,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    generated_text = response
```

##### 方法3: 纯文本生成 (第937-990行)

**适用于**: 纯文本模型或没有图像的情况

```python
else:
    # 使用tokenizer处理文本
    inputs = self.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    # 生成
    outputs = self.model.generate(**inputs, ...)
    
    # 解码（同样只解码新生成的部分）
    generated_text = self.tokenizer.decode(outputs[0][input_length:], ...)
```

#### 4.3 返回结果 (第994-998行)

```python
return {
    "text": generated_text,  # 模型生成的文本答案
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "raw": {"generated_text": generated_text}  # 原始响应（可选）
}
```

---

### 5️⃣ 答案评估：`benchmark.evaluate_answer()`

**文件位置**: `benchmarks/huggingface_benchmark.py` 第797行

**作用**: 将模型答案与标准答案比较，给出正确性和分数

**示例实现**:
```python
def evaluate_answer(self, model_answer, ground_truth, task):
    model_answer_clean = str(model_answer).strip().lower()
    ground_truth_clean = str(ground_truth).strip().lower()
    
    is_correct = model_answer_clean == ground_truth_clean
    score = 1.0 if is_correct else 0.0
    
    return BenchmarkResult(
        task_id=task.task_id,
        question=task.question,
        ground_truth=ground_truth,
        model_answer=model_answer,
        is_correct=is_correct,
        score=score,
        ...
    )
```

---

## 🔄 完整执行示例

假设有一个GQA任务：
- **问题**: "What color is the car?"
- **图像**: 一张包含红色汽车的图片
- **标准答案**: "red"

### 执行流程：

1. **run_benchmark()** 
   - 选择批量处理模式
   - 加载所有任务

2. **_process_task_batch()** - 处理这个任务
   
3. **_build_prompt()**
   - 输入: `task.question = "What color is the car?"`
   - 输出: `prompt = "What color is the car?"`

4. **model_adapter.generate()**
   - 输入图像转换为PIL Image
   - Processor处理: `processor(text="What color is the car?", images=[PIL_Image])`
   - 模型生成: `model.generate(**inputs)` → token IDs
   - 解码: `processor.decode(generated_ids)` → `"red"`
   - 返回: `{"text": "red", "usage": {...}, "raw": {...}}`

5. **提取答案**
   - `model_answer = "red"`

6. **evaluate_answer()**
   - 比较: `"red" == "red"` → `True`
   - 返回: `BenchmarkResult(is_correct=True, score=1.0)`

7. **收集结果**
   - 添加到results列表
   - 返回给调用者

---

## 🎯 关键设计要点

1. **适配器模式**: 使用 `model_adapter` 抽象不同模型的接口
2. **流式处理**: 支持大数据集的流式加载
3. **错误处理**: 单个任务失败不影响其他任务
4. **答案解码**: 只解码新生成的部分，避免重复prompt
5. **可扩展性**: Prompt格式可以根据模型类型调整

---

## 📝 调试输出

在verbose模式下，可以看到：

```
[1/10] 处理任务: n161313...
  📝 Prompt: What color is the car?
  🖼️  图像数量: 1
  🔍 模型返回值类型: <class 'dict'>
  🔍 模型返回值的键: ['text', 'usage', 'raw']
  🔍 模型返回值内容:
    - text: red
    - usage: {'prompt_tokens': 0, 'completion_tokens': 0}
  ✓ 答案: red | GT: red
```

---

## 🔧 常见问题排查

1. **答案重复了问题**: 检查解码逻辑，确保使用 `outputs[0][input_length:]`
2. **图像加载失败**: 检查图像路径格式（URL/Base64/本地路径）
3. **Prompt格式不对**: 修改 `_build_prompt()` 方法
4. **模型调用失败**: 检查模型适配器的初始化参数


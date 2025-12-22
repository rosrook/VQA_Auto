# memory_utils.py 代码详解

## 📋 文件概述

这个文件提供了**内存管理和优化工具**，主要用于：
- 监控系统内存和GPU内存使用情况
- 估算数据集的内存占用
- 推荐数据加载策略
- 计算最优的batch size

---

## 🔍 主要类和功能

### 1. MemoryMonitor（内存监控器）

#### 功能
实时监控和报告内存使用情况

#### 主要方法

##### `get_memory_info()` - 获取内存信息
```python
返回包含以下信息的字典：
- 系统内存：总内存、可用内存、已用内存、使用百分比
- 进程内存：RSS（实际物理内存）、VMS（虚拟内存）
- GPU内存：每个GPU的已分配、已保留、总容量
```

**用途**：在加载数据集或训练前检查可用内存

##### `print_memory_info()` - 打印内存信息
格式化输出内存使用情况，便于查看

##### `check_memory_available(required_gb)` - 检查内存是否足够
- 输入：需要的内存大小（GB）
- 输出：布尔值，表示是否有足够内存
- 如果内存不足会发出警告

##### `clear_memory()` - 清理内存
- 调用Python垃圾回收器
- 清空CUDA缓存
- 用于在训练过程中释放内存

---

### 2. DatasetSizeEstimator（数据集大小估算器）

#### 功能
估算不同类型数据的内存占用

#### 主要方法

##### `estimate_image_memory()` - 估算图像内存
```python
计算方式：
内存(GB) = 图像数量 × 图像尺寸(H×W) × 通道数 × 每元素字节数 / 1024³

支持的dtype：
- float32: 4字节/元素
- float16: 2字节/元素  
- uint8: 1字节/元素
```

**示例**：
- 1000张 224×224 RGB图像 (float32) ≈ 0.18 GB

##### `estimate_text_memory()` - 估算文本内存
```python
计算方式：
内存(GB) = 样本数 × 2 × 最大长度 × 每元素字节数 / 1024³

注意：乘以2是因为需要input_ids和attention_mask两个tensor
```

**示例**：
- 10000个样本，最大长度512 (int64) ≈ 0.08 GB

##### `estimate_vqa_dataset_memory()` - 估算VQA数据集内存
综合估算VQA数据集的内存占用，包括：
- 图像内存（如果预加载）
- 问题文本内存
- 答案文本内存
- 元数据内存

**参数**：
- `preload_images`: 是否预加载图像到内存
  - `True`: 计算图像内存
  - `False`: 图像内存为0（懒加载）

**返回**：包含各部分内存的详细字典

##### `recommend_strategy()` - 推荐数据加载策略
根据数据集大小和可用内存，推荐三种策略：

1. **small_dataset**（小数据集）
   - 条件：所需内存 < 可用内存的50%
   - 策略：可以预加载所有数据

2. **medium_dataset**（中等数据集）
   - 条件：所需内存 < 可用内存的2倍
   - 策略：使用懒加载 + 缓存

3. **large_dataset**（大数据集）
   - 条件：所需内存 ≥ 可用内存的2倍
   - 策略：必须使用流式加载或内存映射

---

### 3. DatasetOptimizer（数据集优化器）

#### 功能
分析数据集并给出具体的优化建议

#### 主要方法

##### `analyze_and_recommend()` - 分析并推荐
**工作流程**：
1. 加载数据集并统计样本数量
2. 获取当前可用内存
3. 估算数据集内存需求
4. 根据策略生成具体建议

**返回的建议包括**：
- 推荐的数据集类型（Dataset/LazyLoadVQADataset/StreamingVQADataset）
- 推荐的batch_size
- 是否启用图像缓存
- num_workers设置
- 是否使用混合精度训练
- 其他优化建议

**示例输出**：
```python
{
    'num_samples': 100000,
    'available_memory_gb': 32.0,
    'estimated_memory_gb': 15.5,
    'recommended_strategy': 'medium_dataset',
    'recommendations': [
        '数据集中等大小，建议使用LazyLoadVQADataset',
        '建议batch_size: 16',
        '启用图像缓存（cache_images=True）',
        ...
    ]
}
```

---

### 4. BatchSizeCalculator（批次大小计算器）

#### 功能
根据GPU内存和模型大小计算最优的batch size

#### 主要方法

##### `calculate_max_batch_size()` - 计算最大batch size
**计算逻辑**：

1. **模型内存**：
   ```
   模型内存(GB) = 参数量(百万) × 4字节 / 1000
   ```

2. **每个样本的激活值内存**：
   ```
   激活值内存 = 图像内存 + 文本特征内存
   - 图像：H × W × 3通道 × 4字节
   - 文本：序列长度 × hidden_size × 4字节
   ```

3. **可用于batch的内存**：
   ```
   可用内存 = (GPU总内存 - 模型内存) × 安全系数(0.7)
   ```

4. **计算batch size**：
   ```
   batch_size = 可用内存 / 每个样本内存
   ```

5. **优化**：向下取整到2的幂（1, 2, 4, 8, 16, 32...）

**为什么取2的幂？**
- GPU并行计算效率更高
- 内存对齐更好
- 训练更稳定

---

## 💡 使用场景

### 场景1：加载数据集前检查内存
```python
# 检查是否有足够内存加载数据集
if not MemoryMonitor.check_memory_available(required_gb=10.0):
    print("内存不足，建议使用流式加载")
```

### 场景2：估算数据集内存需求
```python
# 估算100万样本的VQA数据集内存
estimate = DatasetSizeEstimator.estimate_vqa_dataset_memory(
    num_samples=1000000,
    image_size=(224, 224),
    preload_images=True
)
print(f"需要内存: {estimate['total_gb']:.2f} GB")
```

### 场景3：获取优化建议
```python
# 分析数据集并获取建议
recommendations = DatasetOptimizer.analyze_and_recommend(
    data_path="data/train.json",
    batch_size=16
)
DatasetOptimizer.print_recommendations(recommendations)
```

### 场景4：计算最优batch size
```python
# 计算110M参数模型的最大batch size
max_bs = BatchSizeCalculator.calculate_max_batch_size(
    model_params=110,  # 110M参数
    image_size=(224, 224),
    sequence_length=128,
    available_gpu_memory_gb=16.0
)
print(f"推荐batch size: {max_bs}")
```

---

## 🎯 设计模式

### 1. 静态方法模式
所有类都使用 `@staticmethod`，因为：
- 不需要维护状态
- 方法之间相互独立
- 使用简单，无需实例化

### 2. 分层设计
```
MemoryMonitor (底层监控)
    ↓
DatasetSizeEstimator (中层估算)
    ↓
DatasetOptimizer (高层建议)
```

### 3. 安全系数
在计算batch size时使用 `safety_factor=0.7`，因为：
- 需要为其他操作留出内存
- 避免OOM（内存溢出）错误
- 保证系统稳定性

---

## 🔧 关键计算

### 内存单位转换
```python
1 GB = 1024³ 字节 = 1,073,741,824 字节
```

### 数据类型大小
```python
float32: 4 字节
float16: 2 字节
int64:   8 字节
int32:   4 字节
uint8:   1 字节
```

### 模型参数内存
```python
# 假设使用float32
参数量(GB) = 参数量 × 4字节 / 1024³
```

---

## ⚠️ 注意事项

1. **估算值不精确**：
   - 实际内存使用可能因框架、优化等因素有所不同
   - 建议留出20-30%的余量

2. **GPU内存 vs CPU内存**：
   - 图像数据通常在CPU内存中
   - 模型和激活值在GPU内存中
   - 需要分别考虑

3. **动态内存**：
   - 训练过程中的梯度、优化器状态也会占用内存
   - batch size计算时已考虑安全系数

4. **缓存影响**：
   - 图像缓存会显著增加内存使用
   - 需要权衡速度和内存

---

## 📊 示例输出

运行 `if __name__ == "__main__"` 部分会输出：

```
============================================================
内存使用情况
============================================================
系统总内存: 64.00 GB
系统可用内存: 32.50 GB
系统已用内存: 31.50 GB (49.2%)
进程RSS内存: 2.30 GB
------------------------------------------------------------
GPU 0:
  已分配: 4.20 GB
  已保留: 4.50 GB
  总容量: 16.00 GB
============================================================

============================================================
数据集内存估算
============================================================
样本数量: 100,000
图像内存: 0.00 GB (懒加载)
问题内存: 0.01 GB
答案内存: 0.00 GB
元数据内存: 0.10 GB
总计: 0.11 GB
============================================================

推荐策略: medium_dataset
推荐最大batch size: 16
```

---

## 🎓 总结

这个工具模块提供了**完整的内存管理解决方案**：

1. **监控**：实时了解内存使用情况
2. **估算**：预测数据集和模型的内存需求
3. **建议**：根据实际情况推荐最优策略
4. **优化**：计算最优的batch size

使用这些工具可以：
- ✅ 避免内存溢出错误
- ✅ 优化数据加载策略
- ✅ 提高训练效率
- ✅ 充分利用硬件资源


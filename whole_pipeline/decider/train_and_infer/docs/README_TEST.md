# 测试数据集使用指南

## 快速开始

### 1. 创建测试数据集

```bash
# 方式 1: 使用脚本（推荐）
./create_test_dataset.sh

# 方式 2: 手动运行
python prepare_training_data.py test_input_data.json \
    --output-dir ./test_training_data \
    --train-split 0.7 \
    --val-split 0.2 \
    --test-split 0.1 \
    --save-format huggingface
```

### 2. 查看数据集信息

转换完成后，数据集会保存在 `./test_training_data/` 目录下：

```
test_training_data/
├── train/          # 训练集（约 70%）
├── validation/     # 验证集（约 20%）
├── test/           # 测试集（约 10%）
└── metadata.json   # 数据集元数据
```

### 3. 运行训练测试

```bash
# 使用测试配置文件
python grpo.py --config test_config.yaml

# 或者使用命令行参数覆盖
python grpo.py \
    --config test_config.yaml \
    --model_name_or_path "your-model-path" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1
```

## 测试数据集说明

### 数据规模

- **输入记录数**: 3 条
- **训练样本数**: 4 条（3 个 report × 1-2 个 filtered_datasets）
- **Agent 列表**: `["quality_filter_agent", "deduplication_filter", "length_filter"]`

### 数据特点

1. **Report 1**: 
   - 100 个样本，15% 重复率
   - 8% 信息不完整
   - 1 个 filtered_dataset（14 个样本）

2. **Report 2**:
   - 50 个样本，25% 重复率
   - 图像分辨率低
   - 2 个 filtered_datasets（8 + 5 = 13 个样本）

3. **Report 3**:
   - 30 个样本，5% 重复率
   - 20% 格式问题
   - 1 个 filtered_dataset（11 个样本）

### 数据集划分

- **训练集**: 70% (约 3 个样本)
- **验证集**: 20% (约 1 个样本)
- **测试集**: 10% (约 0-1 个样本)

*注意：由于数据量很小，实际划分可能会有偏差*

## 测试配置说明

`test_config.yaml` 中的关键设置：

### 训练参数（小规模测试）

```yaml
num_train_epochs: 1              # 只训练 1 个 epoch
per_device_train_batch_size: 1  # 批次大小为 1
num_generations: 2              # 每个 prompt 生成 2 个样本
eval_strategy: "no"             # 跳过验证（节省时间）
```

### 奖励函数

```yaml
reward_funcs:
  - "format"           # 格式检查
  - "agent_selection"  # Agent 选择准确性
```

*注意：没有使用 `"combined"` 和 `"filtering_quality"`，因为 `agent_executor` 还未实现*

## 验证训练流程

### 1. 检查数据加载

```python
from datasets import load_from_disk

dataset = load_from_disk("./test_training_data")
print(f"训练集大小: {len(dataset['train'])}")
print(f"验证集大小: {len(dataset['validation'])}")
print(f"测试集大小: {len(dataset['test'])}")

# 查看第一条数据
print("\n第一条训练数据:")
print(dataset['train'][0])
```

### 2. 检查模型加载

确保 `model_name_or_path` 指向一个可用的模型。如果使用本地模型：

```yaml
model_name_or_path: "./path/to/your/model"
```

### 3. 运行小规模训练

```bash
# 使用测试配置
python grpo.py --config test_config.yaml

# 如果遇到内存问题，进一步减小批次大小
python grpo.py --config test_config.yaml \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_generations 2
```

## 常见问题

### Q: 数据集太小，训练会失败吗？

A: 不会。GRPO 算法可以处理小数据集，但：
- 训练效果可能不好（数据量太少）
- 主要用于验证训练流程是否能正常运行
- 建议至少准备 10-20 个训练样本用于实际训练

### Q: 如何增加测试数据？

A: 编辑 `test_input_data.json`，添加更多记录：

```json
{
    "report": "你的报告文本",
    "agent_list": ["quality_filter_agent", "deduplication_filter"],
    "filtered_datasets": [
        {
            "data_ids": [0, 1, 2, ...],
            "metadata": {...}
        }
    ]
}
```

然后重新运行 `create_test_dataset.sh`。

### Q: 训练时出现内存不足？

A: 进一步减小参数：

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
num_generations: 1  # 最小化生成数量
max_prompt_length: 128
max_completion_length: 128
```

### Q: 如何验证训练是否成功？

A: 检查：

1. **输出目录**: `./test_output/` 中应该有模型文件
2. **日志**: 应该看到 loss 和 reward 的日志
3. **没有错误**: 训练应该能完成至少 1 个 epoch

## 下一步

测试成功后，可以：

1. **准备真实数据**: 使用 `prepare_training_data.py` 处理实际数据
2. **调整配置**: 修改 `config.yaml` 使用更大的参数
3. **实现 agent_executor**: 启用 `filtering_quality` 奖励函数
4. **完整训练**: 使用真实数据集进行完整训练


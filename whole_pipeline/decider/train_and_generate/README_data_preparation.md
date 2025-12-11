# 数据准备说明

## 功能

`prepare_training_data.py` 用于将原始数据格式转换为 GRPO 训练所需的数据集格式。

## 输入格式

输入文件可以是 JSON 或 JSONL 格式，包含一个字典列表。每个字典的结构如下：

```json
{
    "report": "数据集报告文本（字符串）",
    "agent_list": ["agent1", "agent2", ...],  // 所有 report 共享同一个 agent list
    "filtered_datasets": [
        {
            "data_ids": [0, 1, 5, 7, ...],  // 筛选后的数据 ID/索引列表
            "metadata": {...}  // 可选的元数据
        },
        ...
    ]  // 与 report 一一对应，每个 filtered_dataset 对应一个训练样本
}
```

### 字段说明

- **report** (必需): 数据集报告文本，描述数据集的特征和质量问题
- **agent_list** (必需): Agent 名称列表，所有 report 共享同一个列表
- **filtered_datasets** (必需): 筛选后的数据集列表，每个元素包含：
  - **data_ids** (必需): 应该保留的数据 ID/索引列表
  - **metadata** (可选): 额外的元数据信息

### 示例

参考 `example_input.json` 文件查看完整示例。

## 输出格式

转换后的数据格式符合 GRPO 训练要求：

```json
{
    "report": "数据集报告文本",
    "ground_truth_agents": ["agent1", "agent2", ...],
    "ground_truth_filtered_data": [0, 1, 5, 7, ...],
    "metadata": {...}  // 如果输入中有 metadata，会保留
}
```

## 使用方法

### 基本用法

```bash
python prepare_training_data.py input.json --output-dir ./training_data
```

### 完整参数

```bash
python prepare_training_data.py input.json \
    --output-dir ./training_data \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1 \
    --save-format huggingface \
    --agent-list agent1 agent2 agent3  # 可选，如果输入中没有 agent_list
```

### 参数说明

- `input_file`: 输入文件路径（JSON 或 JSONL）
- `--output-dir`: 输出目录（默认: `./training_data`）
- `--train-split`: 训练集比例（默认: 0.8）
- `--val-split`: 验证集比例（默认: 0.1）
- `--test-split`: 测试集比例（默认: 0.1）
- `--save-format`: 保存格式，可选 `huggingface`、`json`、`jsonl`（默认: `huggingface`）
- `--agent-list`: 全局 agent 列表（如果输入数据中没有 `agent_list` 字段）
- `--validate-only`: 仅验证输入格式，不进行转换

## 输出文件结构

### HuggingFace 格式（默认）

```
training_data/
├── train/
│   ├── dataset_info.json
│   └── state.json
├── validation/
│   ├── dataset_info.json
│   └── state.json
├── test/
│   ├── dataset_info.json
│   └── state.json
└── metadata.json
```

### JSON/JSONL 格式

```
training_data/
├── train.json (或 train.jsonl)
├── validation.json (或 validation.jsonl)
└── test.json (或 test.jsonl)
```

## 数据转换逻辑

1. **输入验证**: 检查必需字段和数据类型
2. **Agent 列表处理**: 
   - 如果提供了全局 `--agent-list`，使用它
   - 否则从第一条记录的 `agent_list` 获取
   - 如果不同记录的 `agent_list` 不一致，会发出警告
3. **样本展开**: 
   - 每个 `report` + `filtered_dataset` 组合生成一个训练样本
   - 例如：1 个 report + 2 个 filtered_datasets = 2 个训练样本
4. **数据集划分**: 按比例随机划分训练/验证/测试集

## 验证模式

仅验证输入格式，不进行转换：

```bash
python prepare_training_data.py input.json --validate-only
```

## 注意事项

1. **Agent 列表一致性**: 建议所有记录使用相同的 `agent_list`，否则会发出警告
2. **数据 ID 格式**: `data_ids` 可以是整数或字符串，但建议使用整数索引
3. **数据集划分**: 确保 `train-split + val-split + test-split = 1.0`
4. **元数据保留**: 如果输入中有 `metadata` 字段，会被保留在输出中

## 示例工作流

```bash
# 1. 准备输入数据（JSON 格式）
# 参考 example_input.json

# 2. 验证数据格式
python prepare_training_data.py example_input.json --validate-only

# 3. 转换为训练数据集（HuggingFace 格式）
python prepare_training_data.py example_input.json \
    --output-dir ./training_data \
    --save-format huggingface

# 4. 在训练脚本中使用
# 在 config.yaml 中设置：
# dataset_name: "./training_data"
```


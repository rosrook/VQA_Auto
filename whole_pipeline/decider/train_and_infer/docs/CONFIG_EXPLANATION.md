# config.yaml 配置说明

## 概述

`config.yaml` 是 GRPO 训练脚本的配置文件，使用 `TrlParser` 解析，包含三个主要部分的配置参数：

1. **Script Arguments** - 脚本特定的参数（`DataFilteringGRPOScriptArguments`）
2. **Training Arguments** - 训练配置（`GRPOConfig`，继承自 `TrainingArguments`）
3. **Model Arguments** - 模型配置（`ModelConfig`）

## 配置结构详解

### 1. Script Arguments（脚本参数）

这部分参数是专门为数据过滤 agent 选择任务设计的。

#### `dataset_name` (必需)
```yaml
dataset_name: "your/dataset"
```
- **作用**: 指定要使用的数据集名称或路径
- **格式**: 
  - HuggingFace Hub 数据集: `"username/dataset_name"`
  - 本地路径: `"./training_data"` 或 `"/path/to/dataset"`
- **在代码中的使用**: `load_dataset(script_args.dataset_name, name=script_args.dataset_config)`

#### `dataset_config` (可选)
```yaml
dataset_config: null
```
- **作用**: 数据集配置名称（用于有多个配置的数据集）
- **默认值**: `null`
- **示例**: 如果数据集有多个配置，可以指定 `"config1"`

#### `dataset_train_split` / `dataset_test_split`
```yaml
dataset_train_split: "train"
dataset_test_split: "test"
```
- **作用**: 指定训练集和测试集的 split 名称
- **默认值**: `"train"` 和 `"test"`
- **在代码中的使用**: 
  - `dataset[script_args.dataset_train_split]` - 获取训练数据
  - `dataset[script_args.dataset_test_split]` - 获取测试数据

#### `reward_funcs` (重要)
```yaml
reward_funcs:
  - "combined"
  - "format"
```
- **作用**: 指定使用的奖励函数列表
- **可选值**:
  - `"agent_selection"`: Agent 选择准确性奖励（F1 score）
  - `"filtering_quality"`: 过滤质量奖励（F1 score，需要 `agent_executor`）
  - `"format"`: 输出格式正确性奖励（XML 格式检查）
  - `"combined"`: 组合奖励（加权组合 `agent_selection` 和 `filtering_quality`）
- **在代码中的使用**: 
  - 从 `reward_funcs_registry` 中查找对应的奖励函数
  - 多个奖励函数会**相加**（见 `grpo_trainer.py` 第 440 行）
- **建议**: 
  - 如果 `agent_executor` 未实现，使用 `["format", "agent_selection"]`
  - 如果 `agent_executor` 已实现，使用 `["combined", "format"]`

#### `agent_selection_weight` / `filtering_quality_weight`
```yaml
agent_selection_weight: 0.5
filtering_quality_weight: 0.5
```
- **作用**: 当使用 `"combined"` 奖励函数时，这两个权重用于平衡两种奖励
- **默认值**: 都是 `0.5`
- **在代码中的使用**: 
  ```python
  combined_reward(
      completions,
      agent_weight=script_args.agent_selection_weight,  # 0.5
      filtering_weight=script_args.filtering_quality_weight,  # 0.5
      **kwargs
  )
  ```
- **调整建议**: 
  - 如果更关注 agent 选择准确性，增加 `agent_selection_weight`
  - 如果更关注过滤质量，增加 `filtering_quality_weight`

#### `available_agents` (必需)
```yaml
available_agents:
  - "quality_filter"
  - "deduplication_filter"
  - "length_filter"
  - "relevance_filter"
```
- **作用**: 指定可用的 agent 列表，用于生成 system prompt
- **在代码中的使用**: 
  - `create_system_prompt(script_args.available_agents)` - 创建系统提示
  - 模型需要从这个列表中选择合适的 agents
- **重要**: 这个列表必须与 `agents_catalog.json` 中的 agent 名称匹配

---

### 2. Training Arguments（训练参数）

这部分参数继承自 HuggingFace 的 `TrainingArguments` 和 TRL 的 `GRPOConfig`。

#### `output_dir`
```yaml
output_dir: "./output"
```
- **作用**: 模型和检查点的保存目录
- **在代码中的使用**: `trainer.save_model(training_args.output_dir)`

#### `num_train_epochs`
```yaml
num_train_epochs: 3
```
- **作用**: 训练轮数
- **默认值**: 3
- **调整建议**: 
  - 小数据集: 5-10 epochs
  - 大数据集: 1-3 epochs

#### `per_device_train_batch_size`
```yaml
per_device_train_batch_size: 4
```
- **作用**: 每个设备的训练批次大小
- **注意**: GRPO 会为每个 prompt 生成 `num_generations` 个样本，实际内存使用会更大
- **调整建议**: 
  - GPU 内存 24GB: `batch_size=2-4`
  - GPU 内存 40GB+: `batch_size=4-8`

#### `learning_rate`
```yaml
learning_rate: 1e-5
```
- **作用**: 学习率
- **默认值**: `1e-5`
- **调整建议**: 
  - 全量微调: `1e-5` 到 `5e-5`
  - LoRA 微调: `1e-4` 到 `5e-4`

#### 其他可用的 Training Arguments

虽然 `config.yaml` 中没有列出，但你可以添加任何 `TrainingArguments` 支持的参数，例如：

```yaml
# 训练参数扩展示例
gradient_accumulation_steps: 4
warmup_steps: 100
logging_steps: 10
save_steps: 500
eval_strategy: "steps"
eval_steps: 500
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: "reward"
greater_is_better: true
```

#### GRPO 特定的参数

这些参数在 `GRPOConfig` 中定义：

```yaml
# GRPO 特定参数（如果需要在 config.yaml 中设置）
max_prompt_length: 512        # Prompt 最大长度
max_completion_length: 512   # 生成文本最大长度
num_generations: 4           # 每个 prompt 生成的样本数（G in GRPO paper）
temperature: 0.7             # 生成温度
beta: 0.1                    # KL 散度惩罚系数
```

---

### 3. Model Arguments（模型参数）

#### `model_name_or_path` (必需)
```yaml
model_name_or_path: "meta-llama/Llama-2-7b-hf"
```
- **作用**: 指定要使用的模型
- **格式**: 
  - HuggingFace Hub: `"meta-llama/Llama-2-7b-hf"`
  - 本地路径: `"./models/my_model"`
  - Qwen2-VL: `"Qwen/Qwen2-VL-2B-Instruct"`
- **在代码中的使用**: 
  - `GRPOTrainer(model=model_args.model_name_or_path, ...)`
  - 如果包含 `"Qwen2-VL"`，会使用 `Qwen2VLForConditionalGeneration`

#### 其他可用的 Model Arguments

```yaml
# 模型参数扩展示例
use_peft: true              # 是否使用 PEFT（LoRA）
lora_r: 16                   # LoRA rank
lora_alpha: 32               # LoRA alpha
lora_dropout: 0.05           # LoRA dropout
torch_dtype: "float16"       # 模型数据类型
```

---

## 配置文件使用方式

### 方式 1: 通过命令行参数
```bash
python grpo.py --config config.yaml
```

### 方式 2: 通过环境变量
```bash
export CONFIG_FILE=config.yaml
python grpo.py
```

### 方式 3: 命令行参数覆盖配置
```bash
python grpo.py --config config.yaml \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --learning_rate 2e-5 \
    --num_train_epochs 5
```

---

## 完整配置示例

```yaml
# Script arguments
dataset_name: "./training_data"
dataset_config: null
dataset_train_split: "train"
dataset_test_split: "test"
reward_funcs:
  - "combined"
  - "format"
agent_selection_weight: 0.6
filtering_quality_weight: 0.4
available_agents:
  - "quality_filter_agent"
  - "deduplication_filter"
  - "length_filter"
  - "relevance_filter"
  - "alignment_agent"
  - "robustness_agent"

# Training arguments (GRPOConfig)
output_dir: "./output/grpo_model"
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1e-5
warmup_steps: 100
logging_steps: 10
save_steps: 500
eval_strategy: "steps"
eval_steps: 500
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: "reward"
greater_is_better: true

# GRPO 特定参数
max_prompt_length: 512
max_completion_length: 512
num_generations: 4
temperature: 0.7
beta: 0.1

# Model arguments
model_name_or_path: "Qwen/Qwen2-VL-2B-Instruct"
use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
torch_dtype: "float16"
```

---

## 配置验证

运行训练脚本时，`TrlParser` 会自动：
1. 解析 YAML 文件
2. 验证必需参数是否存在
3. 将参数映射到对应的配置类
4. 如果命令行提供了参数，会覆盖 YAML 中的值

如果配置有误，会抛出相应的错误信息。

---

## 常见问题

### Q: 如何知道哪些参数可以在 config.yaml 中设置？
A: 查看对应的配置类定义：
- Script arguments: `DataFilteringGRPOScriptArguments` (在 `grpo.py` 中)
- Training arguments: `GRPOConfig` (在 TRL 库中，继承自 `TrainingArguments`)
- Model arguments: `ModelConfig` (在 TRL 库中)

### Q: 配置参数和命令行参数哪个优先级更高？
A: 命令行参数优先级更高，会覆盖 YAML 文件中的值。

### Q: 如何调试配置问题？
A: 添加 `--validate-only` 参数（如果支持），或者先运行一个小的训练步骤查看错误信息。


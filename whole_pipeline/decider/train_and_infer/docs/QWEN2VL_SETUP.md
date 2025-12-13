# Qwen2-VL 模型配置说明

## 模型信息

- **HuggingFace 模型名称**: `Qwen/Qwen2-VL-7B-Instruct`
- **模型类型**: 视觉语言模型（Vision-Language Model）
- **支持功能**: 文本 + 图像多模态输入

## 配置文件更新

已更新以下配置文件以使用 Qwen2-VL：

1. **`test_config.yaml`** - 测试配置
2. **`config.yaml`** - 主配置

### 关键配置参数

```yaml
# Model arguments
model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
```

**注意**: `torch_dtype` 和 `attn_implementation` 等参数需要通过 `GRPOConfig` 的 `model_init_kwargs` 传递。由于 `TrlParser` 的限制，这些参数可能需要：

1. **通过代码修改**: 在 `GRPOConfig` 中添加 `model_init_kwargs` 字段
2. **使用默认值**: `grpo_trainer.py` 中 `attn_implementation` 默认是 `"flash_attention_2"`
3. **命令行参数**: 如果 TrlParser 支持，可以通过命令行传递

### Qwen2-VL 默认参数

`grpo_trainer.py` 中已设置默认值：
- `attn_implementation`: `"flash_attention_2"`（默认）
- `max_pixels`: `12845056`（默认）
- `min_pixels`: `3136`（默认）
- `torch_dtype`: 需要在模型加载时指定（可通过 `model_init_kwargs`）

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install transformers torch accelerate

# Flash Attention 2（可选，用于加速）
pip install -U flash-attn --no-build-isolation

# 其他依赖
pip install datasets trl peft
```

### 2. 验证模型访问

确保可以访问 HuggingFace Hub：

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 测试加载模型（仅验证，不完整加载）
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    print("✓ 可以访问 Qwen2-VL 模型")
except Exception as e:
    print(f"✗ 无法访问模型: {e}")
    print("  可能需要设置 HuggingFace token:")
    print("  export HF_TOKEN=your_token")
```

### 3. 设置 HuggingFace Token（如果需要）

如果模型是 gated（需要授权访问）：

```bash
# 方式 1: 环境变量
export HF_TOKEN=your_huggingface_token

# 方式 2: 登录
huggingface-cli login
```

## 模型规格

### Qwen2-VL-7B-Instruct

- **参数量**: 7B
- **显存需求**: 
  - FP32: ~28GB
  - BF16: ~14GB
  - INT8: ~7GB（如果使用量化）
- **推荐配置**: 
  - GPU: 至少 24GB 显存（使用 BF16）
  - 推荐: 40GB+ 显存（A100/H100）

### 其他可用模型

如果 7B 模型太大，可以考虑：

- `Qwen/Qwen2-VL-2B-Instruct` - 2B 参数，显存需求更低
- `Qwen/Qwen2-VL-72B-Instruct` - 72B 参数，性能更强

修改 `model_name_or_path` 即可切换。

## 训练配置建议

### 小规模测试（test_config.yaml）

```yaml
per_device_train_batch_size: 1      # 小批次
gradient_accumulation_steps: 4      # 累积梯度
num_generations: 2                  # 减少生成数量
max_prompt_length: 512
max_completion_length: 512
```

### 正式训练（config.yaml）

```yaml
per_device_train_batch_size: 2      # 根据显存调整
gradient_accumulation_steps: 8      # 累积梯度
num_generations: 4                  # 标准生成数量
max_prompt_length: 1024
max_completion_length: 1024
```

## 显存优化建议

### 1. 使用混合精度训练

```yaml
fp16: true  # 或 bf16: true（如果 GPU 支持）
```

### 2. 使用梯度检查点

```yaml
gradient_checkpointing: true
```

### 3. 使用 LoRA（PEFT）

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

### 4. 使用 DeepSpeed ZeRO

如果显存仍然不足，可以使用 DeepSpeed：

```bash
# 安装 DeepSpeed
pip install deepspeed

# 使用 DeepSpeed 配置
deepspeed --num_gpus=1 grpo.py --config config.yaml --deepspeed ds_config.json
```

## 常见问题

### Q: 模型下载失败？

A: 检查：
1. 网络连接是否正常
2. 是否需要 HuggingFace token
3. 磁盘空间是否足够（模型约 14GB）

### Q: 显存不足（OOM）？

A: 尝试：
1. 减小 `per_device_train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 减小 `num_generations`
4. 使用 LoRA（PEFT）
5. 使用更小的模型（2B）

### Q: Flash Attention 2 安装失败？

A: 
- Flash Attention 2 是可选的，不是必需的
- 如果安装失败，可以移除 `attn_implementation: "flash_attention_2"`
- 模型仍可正常训练，只是速度稍慢

### Q: 如何验证模型是否正确加载？

A: 运行快速测试：

```bash
python quick_test.py
```

或手动验证：

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="bfloat16"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print("✓ 模型加载成功")
```

## 下一步

1. **验证环境**: 确保可以加载模型
2. **准备数据**: 使用 `prepare_training_data.py` 准备训练数据
3. **小规模测试**: 使用 `test_config.yaml` 进行测试
4. **正式训练**: 使用 `config.yaml` 进行完整训练

## 参考资源

- [Qwen2-VL 官方文档](https://huggingface.co/docs/transformers/model_doc/qwen2_vl)
- [Qwen2-VL 模型卡片](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Flash Attention 2 文档](https://github.com/Dao-AILab/flash-attention)


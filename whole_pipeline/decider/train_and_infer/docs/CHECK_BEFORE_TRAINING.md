# 训练前检查清单

在运行 `grpo.py` 开始训练之前，请确保完成以下步骤：

## ✅ 必需步骤

### 1. 准备训练数据集

训练数据集还不存在，需要先创建：

```bash
cd /Users/zhuxuzhou/Documents/VQA_Auto/decider

# 方式 1: 使用自动化脚本
./create_test_dataset.sh

# 方式 2: 手动运行
python prepare_training_data.py test_input_data.json \
    --output-dir ./test_training_data \
    --train-split 0.7 \
    --val-split 0.2 \
    --test-split 0.1 \
    --save-format huggingface
```

**验证**: 检查 `./test_training_data/` 目录是否存在，且包含 `train/`, `validation/`, `test/` 子目录。

### 2. 验证配置文件

检查 `test_config.yaml` 中的关键配置：

- ✅ `dataset_name`: `"./test_training_data"` （应该指向训练数据集目录）
- ✅ `model_name_or_path`: `"Qwen/Qwen2-VL-7B-Instruct"` （模型路径）
- ✅ `available_agents`: 包含 agent 列表
- ✅ `output_dir`: `"./test_output"` （输出目录）

### 3. 检查 Python 环境

确保安装了必要的依赖：

```bash
# 检查关键依赖
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import trl; print('trl: OK')"
python -c "import datasets; print('datasets: OK')"
```

如果缺少依赖，安装：

```bash
pip install transformers torch accelerate datasets trl peft
```

### 4. 验证模型访问

确保可以访问 Qwen2-VL 模型（可能需要 HuggingFace token）：

```python
from transformers import AutoProcessor

# 测试访问（不完整加载）
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    print("✓ 可以访问 Qwen2-VL 模型")
except Exception as e:
    print(f"✗ 无法访问模型: {e}")
    print("  可能需要设置 HuggingFace token:")
    print("  export HF_TOKEN=your_token")
    print("  或运行: huggingface-cli login")
```

### 5. 检查显存

Qwen2-VL-7B 需要约 14GB 显存（使用 BF16）。检查可用显存：

```bash
# 如果使用 NVIDIA GPU
nvidia-smi

# 或使用 Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB') for i in range(torch.cuda.device_count())]"
```

如果显存不足，考虑：
- 使用更小的模型：`Qwen/Qwen2-VL-2B-Instruct`
- 减小 `per_device_train_batch_size`
- 使用 LoRA（PEFT）

## 🚀 快速检查脚本

运行以下命令进行快速检查：

```bash
cd /Users/zhuxuzhou/Documents/VQA_Auto/decider

# 运行快速检查
python quick_test.py
```

## 📋 训练前检查清单

- [ ] 训练数据集已创建（`./test_training_data/` 存在）
- [ ] 配置文件已更新（`test_config.yaml`）
- [ ] Python 依赖已安装（transformers, torch, trl 等）
- [ ] 可以访问 Qwen2-VL 模型
- [ ] GPU 显存充足（至少 14GB 推荐）
- [ ] 输出目录可写（`./test_output/`）

## 🎯 开始训练

完成所有检查后，运行：

```bash
cd /Users/zhuxuzhou/Documents/VQA_Auto/decider

# 使用测试配置
python grpo.py --config test_config.yaml

# 如果遇到问题，可以添加调试信息
python grpo.py --config test_config.yaml --logging_level DEBUG
```

## ⚠️ 常见问题

### Q: 训练数据集不存在？

A: 运行 `./create_test_dataset.sh` 或手动运行 `prepare_training_data.py`

### Q: 模型下载失败？

A: 
1. 检查网络连接
2. 设置 HuggingFace token: `export HF_TOKEN=your_token`
3. 或运行: `huggingface-cli login`

### Q: 显存不足（OOM）？

A: 
1. 减小 `per_device_train_batch_size`（在 `test_config.yaml` 中）
2. 增加 `gradient_accumulation_steps`
3. 减小 `num_generations`
4. 使用更小的模型（2B）

### Q: 找不到模块？

A: 安装缺失的依赖：
```bash
pip install transformers torch accelerate datasets trl peft
```

## 📝 下一步

完成检查后，可以：
1. 运行小规模测试训练
2. 检查训练日志和输出
3. 根据结果调整配置
4. 准备真实数据进行完整训练


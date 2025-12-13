#!/bin/bash
# 使用 Qwen2-VL 运行训练的脚本示例

echo "=== 使用 Qwen2-VL 模型训练 ==="

# 检查模型路径
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
echo "模型: $MODEL_PATH"

# 方式 1: 使用配置文件（推荐）
# 注意：torch_dtype 和 attn_implementation 需要通过命令行参数传递
python grpo.py \
    --config test_config.yaml \
    --model_name_or_path "$MODEL_PATH"

# 方式 2: 如果需要传递额外的模型初始化参数
# 注意：这需要 TrlParser 支持 model_init_kwargs，或者修改代码
# python grpo.py \
#     --config test_config.yaml \
#     --model_name_or_path "$MODEL_PATH" \
#     --model_init_kwargs '{"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"}'

echo ""
echo "=== 训练完成 ==="
echo "模型输出目录: ./test_output"


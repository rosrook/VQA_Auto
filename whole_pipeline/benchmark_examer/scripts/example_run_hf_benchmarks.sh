#!/bin/bash
# 示例：运行HuggingFace Benchmark测试

# 1. 列出所有可用的benchmark
python scripts/main.py --list-benchmarks

# 2. 测试指定benchmark（GQA和CLEVR）
python scripts/main.py \
    --model-config configs/example_hf_model_config.json \
    --benchmark-names GQA CLEVR \
    --output results_gqa_clevr.json \
    --max-samples 10

# 3. 根据模型类型和需求筛选benchmark
python scripts/main.py \
    --model-config configs/example_hf_model_config.json \
    --model-type "vision-language" \
    --requirements configs/example_requirements.json \
    --output results_filtered.json

# 4. 测试所有benchmark
python scripts/main.py \
    --model-config configs/example_hf_model_config.json \
    --output results_all.json \
    --max-samples 50

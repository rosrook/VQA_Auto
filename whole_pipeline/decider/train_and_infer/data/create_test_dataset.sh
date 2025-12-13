#!/bin/bash
# 创建测试数据集的脚本

echo "=== 创建测试数据集 ==="

# 1. 验证输入数据格式
echo ""
echo "步骤 1: 验证输入数据格式..."
python prepare_training_data.py test_input_data.json --validate-only

if [ $? -ne 0 ]; then
    echo "错误: 输入数据格式验证失败"
    exit 1
fi

# 2. 转换为训练数据集（HuggingFace 格式）
echo ""
echo "步骤 2: 转换为训练数据集..."
python prepare_training_data.py test_input_data.json \
    --output-dir ./test_training_data \
    --train-split 0.7 \
    --val-split 0.2 \
    --test-split 0.1 \
    --save-format huggingface

if [ $? -ne 0 ]; then
    echo "错误: 数据转换失败"
    exit 1
fi

echo ""
echo "=== 测试数据集创建完成 ==="
echo "数据集位置: ./test_training_data"
echo ""
echo "数据集统计:"
echo "- 训练集: 70%"
echo "- 验证集: 20%"
echo "- 测试集: 10%"
echo ""
echo "下一步: 更新 config.yaml 中的 dataset_name 为 './test_training_data'"


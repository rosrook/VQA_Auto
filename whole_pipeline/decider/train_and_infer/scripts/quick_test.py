#!/usr/bin/env python3
"""
快速测试脚本：验证测试数据集是否可以正常加载和使用
"""

import json
from pathlib import Path
from datasets import load_from_disk, Dataset

def test_input_data():
    """测试输入数据格式"""
    print("=" * 60)
    print("测试 1: 验证输入数据格式")
    print("=" * 60)
    
    input_file = Path("test_input_data.json")
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✓ 成功加载输入文件: {len(data)} 条记录")
    
    # 验证格式
    for i, item in enumerate(data):
        required_keys = ["report", "agent_list", "filtered_datasets"]
        missing = [k for k in required_keys if k not in item]
        if missing:
            print(f"❌ 第 {i+1} 条记录缺少字段: {missing}")
            return False
        
        print(f"  记录 {i+1}: report长度={len(item['report'])}, "
              f"agents={len(item['agent_list'])}, "
              f"filtered_datasets={len(item['filtered_datasets'])}")
    
    print("✓ 输入数据格式验证通过\n")
    return True


def test_training_dataset():
    """测试训练数据集"""
    print("=" * 60)
    print("测试 2: 验证训练数据集")
    print("=" * 60)
    
    dataset_path = Path("test_training_data")
    if not dataset_path.exists():
        print(f"⚠️  训练数据集不存在: {dataset_path}")
        print("   请先运行: python prepare_training_data.py test_input_data.json --output-dir ./test_training_data")
        return False
    
    try:
        dataset = load_from_disk(str(dataset_path))
        print(f"✓ 成功加载训练数据集")
        
        # 检查各个 split
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                split_data = dataset[split_name]
                print(f"  {split_name}: {len(split_data)} 条样本")
                
                # 检查第一条数据的格式
                if len(split_data) > 0:
                    sample = split_data[0]
                    required_keys = ["report", "ground_truth_agents", "ground_truth_filtered_data"]
                    missing = [k for k in required_keys if k not in sample]
                    if missing:
                        print(f"    ❌ 缺少字段: {missing}")
                        return False
                    print(f"    ✓ 格式正确: report长度={len(sample['report'])}, "
                          f"agents={len(sample['ground_truth_agents'])}, "
                          f"data_ids={len(sample['ground_truth_filtered_data'])}")
        
        print("✓ 训练数据集验证通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 加载训练数据集失败: {e}")
        return False


def test_config():
    """测试配置文件"""
    print("=" * 60)
    print("测试 3: 验证配置文件")
    print("=" * 60)
    
    import yaml
    
    config_file = Path("test_config.yaml")
    if not config_file.exists():
        print(f"⚠️  配置文件不存在: {config_file}")
        return False
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("✓ 成功加载配置文件")
    
    # 检查关键字段
    required_sections = {
        "dataset_name": "Script arguments",
        "output_dir": "Training arguments",
        "model_name_or_path": "Model arguments"
    }
    
    for key, section in required_sections.items():
        if key not in config:
            print(f"  ❌ {section} 缺少字段: {key}")
            return False
        print(f"  ✓ {key}: {config[key]}")
    
    print("✓ 配置文件验证通过\n")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("GRPO 测试数据集验证")
    print("=" * 60 + "\n")
    
    results = []
    
    # 测试输入数据
    results.append(("输入数据", test_input_data()))
    
    # 测试训练数据集（如果存在）
    results.append(("训练数据集", test_training_dataset()))
    
    # 测试配置文件
    try:
        import yaml
        results.append(("配置文件", test_config()))
    except ImportError:
        print("⚠️  跳过配置文件测试（需要 pyyaml）")
        results.append(("配置文件", None))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        if result is True:
            print(f"✓ {name}: 通过")
        elif result is False:
            print(f"❌ {name}: 失败")
        else:
            print(f"⚠️  {name}: 跳过")
    
    print("\n" + "=" * 60)
    
    # 给出下一步建议
    all_passed = all(r for _, r in results if r is not None)
    
    if all_passed:
        print("✓ 所有测试通过！可以开始训练了。")
        print("\n运行训练:")
        print("  python grpo.py --config test_config.yaml")
    else:
        print("⚠️  部分测试未通过，请检查上述错误信息。")
        print("\n建议步骤:")
        print("  1. 确保 test_input_data.json 存在且格式正确")
        print("  2. 运行: python prepare_training_data.py test_input_data.json --output-dir ./test_training_data")
        print("  3. 检查 test_config.yaml 中的配置")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


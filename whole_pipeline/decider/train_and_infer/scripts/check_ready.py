#!/usr/bin/env python3
"""
训练前检查脚本：确认是否可以直接开始训练
"""

import os
import sys
from pathlib import Path

def check_mark(condition, message):
    """打印检查结果"""
    status = "✓" if condition else "✗"
    print(f"{status} {message}")
    return condition

def main():
    """主检查函数"""
    print("=" * 60)
    print("GRPO 训练前检查")
    print("=" * 60 + "\n")
    
    results = []
    base_dir = Path(__file__).parent
    
    # 1. 检查输入数据
    print("1. 检查输入数据...")
    input_file = base_dir / "test_input_data.json"
    results.append(check_mark(input_file.exists(), f"输入数据文件: {input_file.name}"))
    
    # 2. 检查训练数据集
    print("\n2. 检查训练数据集...")
    dataset_dir = base_dir / "test_training_data"
    dataset_exists = dataset_dir.exists() and (dataset_dir / "train").exists()
    results.append(check_mark(dataset_exists, f"训练数据集目录: {dataset_dir.name}"))
    
    if not dataset_exists:
        print("   ⚠️  需要先创建训练数据集:")
        print("     运行: ./create_test_dataset.sh")
        print("     或: python prepare_training_data.py test_input_data.json --output-dir ./test_training_data")
    
    # 3. 检查配置文件
    print("\n3. 检查配置文件...")
    config_file = base_dir / "test_config.yaml"
    results.append(check_mark(config_file.exists(), f"配置文件: {config_file.name}"))
    
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
            has_model = "Qwen/Qwen2-VL" in content or "model_name_or_path" in content
            has_dataset = 'dataset_name: "./test_training_data"' in content
            results.append(check_mark(has_model, "配置文件包含模型路径"))
            results.append(check_mark(has_dataset, "配置文件包含数据集路径"))
    
    # 4. 检查 Python 依赖
    print("\n4. 检查 Python 依赖...")
    dependencies = {
        "transformers": "transformers",
        "torch": "torch",
        "trl": "trl",
        "datasets": "datasets",
        "peft": "peft",
    }
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            results.append(check_mark(True, f"{name} 已安装"))
        except ImportError:
            results.append(check_mark(False, f"{name} 未安装"))
    
    # 5. 检查训练脚本
    print("\n5. 检查训练脚本...")
    grpo_file = base_dir / "grpo.py"
    results.append(check_mark(grpo_file.exists(), f"训练脚本: {grpo_file.name}"))
    
    # 6. 检查输出目录权限
    print("\n6. 检查输出目录...")
    output_dir = base_dir / "test_output"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # 测试写入权限
        test_file = output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        results.append(check_mark(True, f"输出目录可写: {output_dir.name}"))
    except Exception as e:
        results.append(check_mark(False, f"输出目录不可写: {e}"))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ 所有检查通过 ({passed}/{total})")
        print("\n可以开始训练了！运行:")
        print("  python grpo.py --config test_config.yaml")
    else:
        print(f"⚠️  部分检查未通过 ({passed}/{total})")
        print("\n请先完成以下步骤:")
        
        if not dataset_exists:
            print("  1. 创建训练数据集:")
            print("     ./create_test_dataset.sh")
        
        missing_deps = []
        for i, (name, module) in enumerate(dependencies.items(), start=len(results)-len(dependencies)):
            if not results[i]:
                missing_deps.append(name)
        
        if missing_deps:
            print(f"  2. 安装缺失的依赖:")
            print(f"     pip install {' '.join(missing_deps)}")
    
    print("=" * 60 + "\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
从 HuggingFace 拉取模型并自动注册为初版
"""

import sys
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from version.version_manager import VersionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 拉取模型并自动注册为初版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 拉取 Qwen2-VL 模型并注册为初版
  python download_and_register_model.py --model-id Qwen/Qwen2-VL-7B-Instruct

  # 指定版本号
  python download_and_register_model.py --model-id Qwen/Qwen2-VL-7B-Instruct --version v0_base_qwen2vl

  # 指定描述
  python download_and_register_model.py --model-id Qwen/Qwen2-VL-7B-Instruct --description "Qwen2-VL 7B 基础模型"
        """
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace 模型 ID（例如: Qwen/Qwen2-VL-7B-Instruct）"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="版本号（默认: v0_base 或自动生成）"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="版本描述（默认: '基础模型 - {model_id}'）"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace 缓存目录（默认: 使用系统默认缓存）"
    )
    parser.add_argument(
        "--registry-file",
        type=str,
        default="model_registry.json",
        help="版本注册表文件路径（默认: model_registry.json）"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./model_versions",
        help="模型版本存储目录（默认: ./model_versions）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型拉取和注册")
    print("=" * 60)
    print(f"模型 ID: {args.model_id}")
    if args.version:
        print(f"版本号: {args.version}")
    if args.description:
        print(f"描述: {args.description}")
    print("=" * 60)
    
    # 初始化版本管理器
    try:
        manager = VersionManager(
            registry_file=args.registry_file,
            base_dir=args.base_dir
        )
    except Exception as e:
        logger.error(f"初始化版本管理器失败: {e}")
        return 1
    
    # 拉取并注册模型
    try:
        version = manager.register_base_model(
            model_id=args.model_id,
            version=args.version,
            description=args.description,
            cache_dir=args.cache_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ 模型拉取和注册成功！")
        print("=" * 60)
        print(f"版本号: {version}")
        print(f"\n现在可以使用以下命令进行推理:")
        print(f"  python inference/run_inference.py --version {version} \\")
        print(f"      --report \"数据集报告...\" \\")
        print(f"      --available-agents agent1 agent2")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"模型拉取或注册失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


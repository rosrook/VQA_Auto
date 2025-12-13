#!/usr/bin/env python3
"""
简化的推理运行脚本
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference_api import ModelInference
import json


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行模型推理")
    parser.add_argument("--version", type=str, help="模型版本号（例如: v20241212_001）")
    parser.add_argument("--model-path", type=str, help="直接指定模型路径")
    parser.add_argument("--report", type=str, required=True,
                       help="数据集报告文本")
    parser.add_argument("--available-agents", type=str, nargs="+", required=True,
                       help="可用的agent列表")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型推理")
    print("=" * 60)
    
    # 初始化推理接口
    try:
        print(f"\n加载模型...")
        if args.version:
            print(f"版本: {args.version}")
        elif args.model_path:
            print(f"模型路径: {args.model_path}")
        else:
            print("使用最新版本")
        
        inference = ModelInference(
            version=args.version,
            model_path=args.model_path
        )
        print("✓ 模型加载完成\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return 1
    
    # 运行推理
    try:
        print("生成预测...")
        result = inference.predict_agent_selection(
            report=args.report,
            available_agents=args.available_agents,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


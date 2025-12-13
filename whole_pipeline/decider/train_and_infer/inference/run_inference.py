# #!/usr/bin/env python3
# """
# 简化的推理运行脚本
# """

# import sys
# from pathlib import Path

# # 添加父目录到路径
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from inference.inference_api import ModelInference
# import json


# def main():
#     """主函数"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="运行模型推理")
#     parser.add_argument("--version", type=str, help="模型版本号（例如: v20241212_001）")
#     parser.add_argument("--model-path", type=str, help="直接指定模型路径")
#     parser.add_argument("--report", type=str, required=True,
#                        help="数据集报告文本")
#     parser.add_argument("--available-agents", type=str, nargs="+", required=True,
#                        help="可用的agent列表")
#     parser.add_argument("--prompt-mode", type=str, default="system_prompt",
#                        choices=["system_prompt", "direct"],
#                        help="Prompt模式: 'system_prompt' (使用系统提示, 默认) 或 'direct' (直接输入report)")
#     parser.add_argument("--max-tokens", type=int, default=512,
#                        help="最大生成token数")
#     parser.add_argument("--temperature", type=float, default=0.7,
#                        help="温度参数")
    
#     args = parser.parse_args()
    
#     print("=" * 60)
#     print("模型推理")
#     print("=" * 60)
    
#     # 初始化推理接口
#     try:
#         print(f"\n加载模型...")
#         if args.version:
#             print(f"版本: {args.version}")
#         elif args.model_path:
#             print(f"模型路径: {args.model_path}")
#         else:
#             print("使用最新版本")
        
#         inference = ModelInference(
#             version=args.version,
#             model_path=args.model_path
#         )
#         print("✓ 模型加载完成\n")
#     except Exception as e:
#         print(f"✗ 模型加载失败: {e}")
#         return 1
    
#     with open("/home/zhuxuzhou/VQA_Auto/whole_pipeline/data/e_aggregated_data/decision_prompt.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     # 运行推理
#     try:
#         print("生成预测...")
#         print(f"Prompt模式: {args.prompt_mode}")
#         result = inference.predict_agent_selection(
#             report=text,
#             available_agents=args.available_agents,
#             prompt_mode=args.prompt_mode,
#             max_new_tokens=args.max_tokens,
#             temperature=args.temperature
#         )
        
#         print("\n" + "=" * 60)
#         print("预测结果")
#         print("=" * 60)
#         print(json.dumps(result, ensure_ascii=False, indent=2))
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"✗ 推理失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())


#!/usr/bin/env python3
"""
简化的推理运行脚本
"""

import sys
from pathlib import Path
from datetime import datetime
import re

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference_api import ModelInference
import json


def clean_escape_sequences(data):
    """
    递归处理数据中的转义字符，使其更易读
    
    将字符串中的 \n, \t 等转义字符转换为实际的换行和制表符
    但保持 JSON 格式的有效性
    """
    if isinstance(data, dict):
        return {key: clean_escape_sequences(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_escape_sequences(item) for item in data]
    elif isinstance(data, str):
        # 处理常见的转义字符
        # 注意：在 JSON 中，\n 会被正确保存为换行符
        # 但我们需要确保字符串中的字面量 \n 也被处理
        cleaned = data
        # 将字面量的 \n 转换为实际换行（如果它们不是真正的转义序列）
        # 这里我们保持原样，因为 JSON 会自动处理转义序列
        # 但如果用户想要更易读的格式，我们可以提供一个选项
        return cleaned
    else:
        return data


def format_json_for_readability(data):
    """
    格式化 JSON 数据，使其更易读
    将字符串中的转义字符转换为实际字符（用于显示）
    """
    if isinstance(data, dict):
        return {key: format_json_for_readability(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [format_json_for_readability(item) for item in data]
    elif isinstance(data, str):
        # 保持字符串原样，JSON 会自动处理转义
        return data
    else:
        return data


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
    parser.add_argument("--prompt-mode", type=str, default="system_prompt",
                       choices=["system_prompt", "direct"],
                       help="Prompt模式: 'system_prompt' (使用系统提示, 默认) 或 'direct' (直接输入report)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                       help="最大生成token数（默认: 2048，避免截断）")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--output", type=str, default=None,
                       help="保存输出到文件路径（JSON格式）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="保存输出到目录（自动生成文件名）")
    
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

    with open("/home/zhuxuzhou/VQA_Auto/whole_pipeline/data/e_aggregated_data/decision_prompt.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # 运行推理
    try:
        print("生成预测...")
        print(f"Prompt模式: {args.prompt_mode}")
        result = inference.predict_agent_selection(
            report=text,
            available_agents=args.available_agents,
            prompt_mode=args.prompt_mode,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)
        
        # 保存输出到文件
        output_path = None
        if args.output:
            output_path = Path(args.output)
        elif args.output_dir:
            # 自动生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_result_{timestamp}.json"
            output_path = Path(args.output_dir) / filename
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备保存的数据
            save_data = {
                **result,
                "timestamp": datetime.now().isoformat(),
                "model_version": args.version or "latest",
                "report": args.report,
                "available_agents": args.available_agents,
                "prompt_mode": args.prompt_mode,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            
            # 清理和格式化数据（处理转义字符）
            save_data = format_json_for_readability(save_data)
            
            # 保存 JSON 文件
            # 使用 ensure_ascii=False 以支持 Unicode 字符
            # indent=2 使 JSON 格式化，更易读
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 同时保存一个更易读的文本版本（可选）
            # 将 JSON 中的字符串字段提取出来，转换为更易读的格式
            text_output_path = output_path.with_suffix('.txt')
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("推理结果\n")
                f.write("=" * 60 + "\n\n")
                
                if "agents" in save_data:
                    f.write("选择的 Agents:\n")
                    for agent in save_data.get("agents", []):
                        f.write(f"  - {agent}\n")
                    f.write("\n")
                
                if "prompts" in save_data:
                    f.write("Agent Prompts:\n")
                    for agent, prompt in save_data.get("prompts", {}).items():
                        f.write(f"\n{agent}:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"{prompt}\n")
                    f.write("\n")
                
                if "reasoning" in save_data:
                    f.write("推理过程:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{save_data.get('reasoning', '')}\n\n")
                
                if "raw_output" in save_data:
                    f.write("原始输出:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{save_data.get('raw_output', '')}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("元数据\n")
                f.write("=" * 60 + "\n")
                f.write(f"时间戳: {save_data.get('timestamp', '')}\n")
                f.write(f"模型版本: {save_data.get('model_version', '')}\n")
                f.write(f"Prompt 模式: {save_data.get('prompt_mode', '')}\n")
                f.write(f"最大 Tokens: {save_data.get('max_tokens', '')}\n")
                f.write(f"温度: {save_data.get('temperature', '')}\n")
            
            print(f"\n✓ 输出已保存到:")
            print(f"  JSON: {output_path}")
            print(f"  文本: {text_output_path}")
            
            print(f"\n✓ 输出已保存到: {output_path}")
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


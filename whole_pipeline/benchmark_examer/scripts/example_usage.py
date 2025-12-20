#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark测试系统使用示例
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from core.model_adapter import BaseModelAdapter, ModelAdapterFactory
from core.benchmark_manager import BenchmarkManager
from core.test_executor import TestExecutor
from core.result_analyzer import ResultAnalyzer
from benchmarks.example_vqa_benchmark import ExampleVQABenchmark


def example_custom_model_adapter():
    """示例：自定义模型适配器"""
    
    class MyCustomModelAdapter(BaseModelAdapter):
        """自定义模型适配器示例"""
        
        def __init__(self, model_path: str):
            self.model_path = model_path
            # 在这里初始化你的模型
            # self.model = load_your_model(model_path)
        
        def generate(self, prompt: str, images=None, **kwargs):
            """生成模型响应"""
            # 调用你的模型
            # result = self.model.generate(prompt, images)
            result = "示例输出"  # 替换为实际模型输出
            return {
                "text": result,
                "usage": {},
                "raw": {}
            }
        
        def get_model_info(self):
            return {
                "name": "MyCustomModel",
                "type": "custom",
                "model_path": self.model_path
            }
    
    # 注册适配器
    ModelAdapterFactory.register("my_custom", MyCustomModelAdapter)
    
    # 使用
    adapter = ModelAdapterFactory.create("my_custom", model_path="path/to/model")
    return adapter


def example_run_test():
    """示例：运行测试"""
    
    # 方式1: 使用HuggingFace本地模型（推荐）
    from core.model_adapter import HuggingFaceAdapter
    model_adapter = HuggingFaceAdapter(
        model_id="Qwen/Qwen-VL-Chat",
        device="cuda",
        trust_remote_code=True
    )
    
    # 方式2: 使用OpenAI API
    # from core.model_adapter import OpenAIAdapter
    # model_adapter = OpenAIAdapter(
    #     api_key="your-api-key",
    #     base_url="https://api.openai.com/v1",
    #     model="gpt-4-vision-preview"
    # )
    
    # 方式3: 使用HuggingFace Hub API
    # from core.model_adapter import HuggingFaceHubAdapter
    # model_adapter = HuggingFaceHubAdapter(
    #     model_id="Qwen/Qwen-VL-Chat",
    #     api_token="your-hf-token"
    # )
    
    # 2. 创建benchmark
    benchmark = ExampleVQABenchmark(
        name="test_benchmark",
        data_path="data/test_vqa.json"
    )
    
    # 3. 执行测试
    executor = TestExecutor(model_adapter)
    results = executor.run_benchmark(benchmark, max_samples=10)
    
    # 4. 分析结果
    analyzer = ResultAnalyzer(accuracy_threshold=0.7)
    summary = analyzer.analyze_results("test_benchmark", results)
    
    print(f"准确率: {summary.accuracy:.2%}")
    print(f"失败案例数: {len(summary.failed_cases)}")
    
    return summary


if __name__ == "__main__":
    print("Benchmark测试系统使用示例")
    print("请查看 README.md 和 QUICK_START.md 了解详细使用方法")

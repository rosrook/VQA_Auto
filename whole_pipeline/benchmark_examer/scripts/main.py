#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark测试系统主入口
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark测试系统主入口
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from ..core.model_adapter import BaseModelAdapter, ModelAdapterFactory
from ..core.benchmark_manager import BenchmarkManager
from ..core.test_executor import TestExecutor
from ..core.result_analyzer import ResultAnalyzer


def load_model_config(config_path: str) -> Dict[str, Any]:
    """加载模型配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_model_adapter(config: Dict[str, Any]) -> BaseModelAdapter:
    """根据配置创建模型适配器"""
    adapter_type = config.get("type", "openai")
    params = config.get("params", {})
    
    # 如果是自定义适配器，需要先注册
    if "custom_adapter_class" in config:
        adapter_class = config["custom_adapter_class"]
        ModelAdapterFactory.register(adapter_type, adapter_class)
    
    return ModelAdapterFactory.create(adapter_type, **params)


def main():
    parser = argparse.ArgumentParser(description="Benchmark测试系统")
    parser.add_argument("--model-config", type=str, required=True,
                       help="模型配置文件路径（JSON格式）")
    parser.add_argument("--model-type", type=str, default=None,
                       help="模型类型（用于筛选benchmark，可选）")
    parser.add_argument("--benchmark-config", type=str, default=None,
                       help="Benchmark配置文件路径（可选，用于指定要测试的benchmark）")
    parser.add_argument("--benchmark-names", type=str, nargs="+", default=None,
                       help="指定要测试的benchmark名称列表（如: GQA CLEVR）")
    parser.add_argument("--taxonomy-file", type=str, default=None,
                       help="Benchmark分类文件路径（默认使用available_benchmarks_with_internal_taxonomy.json）")
    parser.add_argument("--requirements", type=str, default=None,
                       help="需求配置文件路径（JSON格式，用于自动选择benchmark）")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="输出结果文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="每个benchmark最大测试样本数（用于快速测试）")
    parser.add_argument("--accuracy-threshold", type=float, default=0.7,
                       help="准确率阈值，低于此值认为表现不好")
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="列出所有可用的benchmark并退出")
    
    args = parser.parse_args()
    
    # 加载模型配置
    print("="*60)
    print("加载模型配置...")
    print("="*60)
    model_config = load_model_config(args.model_config)
    model_adapter = create_model_adapter(model_config)
    model_info = model_adapter.get_model_info()
    print(f"✓ 模型: {model_info.get('name')} ({model_info.get('type')})")
    
    # 初始化Benchmark管理器
    print("\n" + "="*60)
    print("初始化Benchmark管理器...")
    print("="*60)
    
    taxonomy_file = args.taxonomy_file
    if not taxonomy_file:
        # 默认路径
        taxonomy_file = str(Path(__file__).parent.parent / "benchmarks" / "available_benchmarks_with_internal_taxonomy.json")
    
    benchmark_manager = BenchmarkManager(
        taxonomy_file=taxonomy_file,
        model_type=args.model_type
    )
    
    # 列出所有可用benchmark
    if args.list_benchmarks:
        print("\n可用Benchmark列表:")
        print("="*60)
        available = benchmark_manager.list_available_benchmarks()
        for bench in available:
            print(f"\n名称: {bench['name']}")
            print(f"  HuggingFace ID: {bench['hf_id']}")
            print(f"  配置: {bench.get('config', 'default')}")
            print(f"  默认Split: {bench.get('default_split', 'N/A')}")
            print(f"  可用Splits: {', '.join(bench.get('available_splits', []))}")
            print(f"  分类字段: {', '.join(bench.get('native_taxonomy_fields', []))}")
        return
    
    # 根据需求选择benchmark
    requirements = None
    if args.requirements:
        with open(args.requirements, 'r', encoding='utf-8') as f:
            requirements = json.load(f)
        print(f"根据需求选择benchmark: {requirements}")
    
    # 如果指定了benchmark名称，使用它们；否则根据需求筛选
    selected_benchmarks = benchmark_manager.select_benchmarks(
        requirements=requirements,
        benchmark_names=args.benchmark_names
    )
    
    if not selected_benchmarks:
        print("警告: 没有找到合适的benchmark")
        return
    
    print(f"✓ 选择了 {len(selected_benchmarks)} 个benchmark")
    for bench in selected_benchmarks:
        info = bench.get_info()
        print(f"  - {info['name']}: {info['num_tasks']} 个任务")
    
    # 执行测试
    print("\n" + "="*60)
    print("开始执行测试...")
    print("="*60)
    
    test_executor = TestExecutor(model_adapter)
    result_analyzer = ResultAnalyzer(accuracy_threshold=args.accuracy_threshold)
    
    all_results = []
    
    for benchmark in selected_benchmarks:
        print(f"\n测试 Benchmark: {benchmark.name}")
        results = test_executor.run_benchmark(
            benchmark=benchmark,
            max_samples=args.max_samples,
            verbose=True
        )
        
        # 分析结果
        summary = result_analyzer.analyze_results(benchmark.name, results)
        
        all_results.append({
            "benchmark_name": summary.benchmark_name,
            "total_tasks": summary.total_tasks,
            "correct_count": summary.correct_count,
            "accuracy": summary.accuracy,
            "average_score": summary.average_score,
            "failed_cases": summary.failed_cases,
            "all_results": [
                {
                    "task_id": r.task_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "model_answer": r.model_answer,
                    "is_correct": r.is_correct,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        })
    
    # 识别表现不好的benchmark
    summaries = [
        ResultAnalyzer.BenchmarkSummary(
            benchmark_name=r["benchmark_name"],
            total_tasks=r["total_tasks"],
            correct_count=r["correct_count"],
            accuracy=r["accuracy"],
            average_score=r["average_score"],
            failed_cases=r["failed_cases"]
        )
        for r in all_results
    ]
    
    poor_performers = result_analyzer.identify_poor_performers(summaries)
    
    # 整理最终结果
    # 格式：每个字典是一个bench的测试结果和模型表现不好的bench案例
    output_data = []
    
    for result in all_results:
        bench_result = {
            "benchmark_name": result["benchmark_name"],
            "total_tasks": result["total_tasks"],
            "correct_count": result["correct_count"],
            "accuracy": result["accuracy"],
            "average_score": result["average_score"],
            "is_poor_performer": result["benchmark_name"] in poor_performers,
            "failed_cases": result["failed_cases"],  # 表现不好的案例
            "all_results": result["all_results"]
        }
        output_data.append(bench_result)
    
    # 添加总体摘要
    summary_data = {
        "model_info": model_info,
        "summary": {
            "total_benchmarks": len(all_results),
            "poor_performers": poor_performers,
            "overall_accuracy": sum(s.accuracy for s in summaries) / len(summaries) if summaries else 0.0
        },
        "benchmark_results": output_data
    }
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"结果保存到: {args.output}")
    print(f"\n总体统计:")
    print(f"  测试的Benchmark数量: {len(all_results)}")
    print(f"  表现不好的Benchmark: {len(poor_performers)}")
    if poor_performers:
        print(f"  表现不好的Benchmark列表: {', '.join(poor_performers)}")
    print(f"  平均准确率: {summary_info['summary']['overall_accuracy']:.2%}")
    
    # 显示每个benchmark的摘要
    print(f"\n各Benchmark结果:")
    for result in all_results:
        print(f"  {result['benchmark_name']}:")
        print(f"    准确率: {result['accuracy']:.2%}")
        print(f"    失败案例数: {len(result['failed_cases'])}")


if __name__ == "__main__":
    main()





# python scripts/main.py \
#     --model-config configs/example_hf_model_config.json \
#     --benchmark-names GQA \
#     --output results.json
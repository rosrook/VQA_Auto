#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析器：分析测试结果，找出表现不好的任务
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from ..benchmarks.base_benchmark import BenchmarkResult


@dataclass
class BenchmarkSummary:
    """Benchmark测试摘要"""
    benchmark_name: str
    total_tasks: int
    correct_count: int
    accuracy: float
    average_score: float
    failed_cases: List[Dict[str, Any]]  # 失败案例详情


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, 
                 accuracy_threshold: float = 0.7,
                 score_threshold: float = 0.5):
        """
        Args:
            accuracy_threshold: 准确率阈值，低于此值认为表现不好
            score_threshold: 分数阈值，低于此值的任务视为失败案例
        """
        self.accuracy_threshold = accuracy_threshold
        self.score_threshold = score_threshold
    
    def analyze_results(self, 
                       benchmark_name: str,
                       results: List[BenchmarkResult]) -> BenchmarkSummary:
        """
        分析单个benchmark的结果
        
        Args:
            benchmark_name: Benchmark名称
            results: 测试结果列表
        
        Returns:
            BenchmarkSummary
        """
        total = len(results)
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / total if total > 0 else 0.0
        
        scores = [r.score for r in results if r.score is not None]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # 找出失败案例（分数低于阈值或答案错误）
        failed_cases = []
        for result in results:
            if not result.is_correct or (result.score is not None and result.score < self.score_threshold):
                failed_cases.append({
                    "task_id": result.task_id,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "model_answer": result.model_answer,
                    "score": result.score,
                    "metadata": result.metadata
                })
        
        return BenchmarkSummary(
            benchmark_name=benchmark_name,
            total_tasks=total,
            correct_count=correct_count,
            accuracy=accuracy,
            average_score=average_score,
            failed_cases=failed_cases
        )
    
    def identify_poor_performers(self, 
                                summaries: List[BenchmarkSummary]) -> List[str]:
        """
        识别表现不好的benchmark
        
        Args:
            summaries: Benchmark摘要列表
        
        Returns:
            表现不好的benchmark名称列表
        """
        poor_performers = []
        for summary in summaries:
            if summary.accuracy < self.accuracy_threshold:
                poor_performers.append(summary.benchmark_name)
        return poor_performers
    
    def generate_report_data(self, 
                            summaries: List[BenchmarkSummary]) -> Dict[str, Any]:
        """
        生成报告数据
        
        Args:
            summaries: Benchmark摘要列表
        
        Returns:
            报告数据字典
        """
        poor_performers = self.identify_poor_performers(summaries)
        
        report = {
            "total_benchmarks": len(summaries),
            "poor_performers": poor_performers,
            "overall_accuracy": sum(s.accuracy for s in summaries) / len(summaries) if summaries else 0.0,
            "benchmark_details": [
                {
                    "benchmark_name": s.benchmark_name,
                    "accuracy": s.accuracy,
                    "average_score": s.average_score,
                    "failed_cases_count": len(s.failed_cases),
                    "failed_cases": s.failed_cases
                }
                for s in summaries
            ]
        }
        
        return report

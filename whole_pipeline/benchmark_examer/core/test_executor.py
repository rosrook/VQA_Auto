#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试执行器：执行benchmark测试
"""

import time
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度条替代
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from .model_adapter import BaseModelAdapter
from ..benchmarks.base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


class TestExecutor:
    """测试执行器"""
    
    def __init__(self, model_adapter: BaseModelAdapter):
        self.model_adapter = model_adapter
        self.model_info = model_adapter.get_model_info()
    
    def run_benchmark(self, 
                     benchmark: BaseBenchmark,
                     max_samples: Optional[int] = None,
                     verbose: bool = True) -> List[BenchmarkResult]:
        """
        运行单个benchmark测试
        
        Args:
            benchmark: Benchmark实例
            max_samples: 最大测试样本数（None表示全部）
            verbose: 是否显示进度
        
        Returns:
            测试结果列表
        """
        tasks = benchmark.get_tasks()
        if max_samples:
            tasks = tasks[:max_samples]
        
        results = []
        
        if verbose:
            tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}")
        else:
            tasks_iter = tasks
        
        for task in tasks_iter:
            try:
                # 构建提示
                prompt = self._build_prompt(task)
                
                # 调用模型
                model_response = self.model_adapter.generate(
                    prompt=prompt,
                    images=task.images
                )
                
                # 评估答案
                result = benchmark.evaluate_answer(
                    model_answer=model_response.get("text", ""),
                    ground_truth=task.ground_truth,
                    task=task
                )
                
                results.append(result)
                
            except Exception as e:
                # 记录错误
                result = BenchmarkResult(
                    task_id=task.task_id,
                    question=task.question,
                    ground_truth=task.ground_truth,
                    model_answer="",
                    is_correct=False,
                    score=0.0,
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _build_prompt(self, task: BenchmarkTask) -> str:
        """构建提示"""
        # 基本提示格式
        prompt = task.question
        return prompt

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 测试执行器：执行benchmark测试
# """

# import time
# from typing import List, Dict, Any, Optional

# try:
#     from tqdm import tqdm
#     HAS_TQDM = True
# except ImportError:
#     HAS_TQDM = False
#     # 简单的进度条替代
#     def tqdm(iterable, desc=""):
#         print(f"{desc}...")
#         return iterable

# import sys
# from pathlib import Path

# # 添加项目根目录到路径
# current_dir = Path(__file__).parent.parent
# sys.path.insert(0, str(current_dir))

# from .model_adapter import BaseModelAdapter
# from ..benchmarks.base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


# class TestExecutor:
#     """测试执行器"""
    
#     def __init__(self, model_adapter: BaseModelAdapter):
#         self.model_adapter = model_adapter
#         self.model_info = model_adapter.get_model_info()
    
#     def run_benchmark(self, 
#                      benchmark: BaseBenchmark,
#                      max_samples: Optional[int] = None,
#                      verbose: bool = True) -> List[BenchmarkResult]:
#         """
#         运行单个benchmark测试
        
#         Args:
#             benchmark: Benchmark实例
#             max_samples: 最大测试样本数（None表示全部）
#             verbose: 是否显示进度
        
#         Returns:
#             测试结果列表
#         """
#         tasks = benchmark.get_tasks()
#         if max_samples:
#             tasks = tasks[:max_samples]
        
#         results = []
        
#         if verbose:
#             tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}")
#         else:
#             tasks_iter = tasks
        
#         for task in tasks_iter:
#             try:
#                 # 构建提示
#                 prompt = self._build_prompt(task)
                
#                 # 调用模型
#                 model_response = self.model_adapter.generate(
#                     prompt=prompt,
#                     images=task.images
#                 )
                
#                 # 评估答案
#                 result = benchmark.evaluate_answer(
#                     model_answer=model_response.get("text", ""),
#                     ground_truth=task.ground_truth,
#                     task=task
#                 )
                
#                 results.append(result)
                
#             except Exception as e:
#                 # 记录错误
#                 result = BenchmarkResult(
#                     task_id=task.task_id,
#                     question=task.question,
#                     ground_truth=task.ground_truth,
#                     model_answer="",
#                     is_correct=False,
#                     score=0.0,
#                     metadata={"error": str(e)}
#                 )
#                 results.append(result)
        
#         return results
    
#     def _build_prompt(self, task: BenchmarkTask) -> str:
#         """构建提示"""
#         # 基本提示格式
#         prompt = task.question
#         return prompt







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
                     batch_size: int = 1,
                     verbose: bool = True) -> List[BenchmarkResult]:
        """
        运行单个benchmark测试（支持流式和批量处理）
        
        Args:
            benchmark: Benchmark实例
            max_samples: 最大测试样本数（None表示全部）
            batch_size: 批处理大小（1表示逐个处理，>1表示批量处理）
            verbose: 是否显示进度
        
        Returns:
            测试结果列表
        """
        # 检查是否支持流式加载
        use_streaming = hasattr(benchmark, 'get_dataset_iterator') and hasattr(benchmark, 'get_task_from_item')
        
        if use_streaming and hasattr(benchmark, '_use_streaming') and benchmark._use_streaming:
            # 流式处理模式
            results = self._run_benchmark_streaming(
                benchmark, max_samples, batch_size, verbose
            )
        else:
            # 普通模式（一次性加载所有任务）
            tasks = benchmark.get_tasks()
            if max_samples:
                tasks = tasks[:max_samples]
            
            if verbose:
                tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}")
            else:
                tasks_iter = tasks
            
            results = self._process_tasks(benchmark, tasks_iter, batch_size)
        
        return results
    
    def _run_benchmark_streaming(self,
                                benchmark: BaseBenchmark,
                                max_samples: Optional[int],
                                batch_size: int,
                                verbose: bool) -> List[BenchmarkResult]:
        """流式处理benchmark"""
        results = []
        dataset_iter = benchmark.get_dataset_iterator()
        
        processed_count = 0
        
        # 创建进度条
        if verbose:
            if max_samples:
                pbar = tqdm(total=max_samples, desc=f"Testing {benchmark.name} (streaming)")
            else:
                pbar = tqdm(desc=f"Testing {benchmark.name} (streaming)")
        
        try:
            batch_tasks = []
            batch_indices = []
            
            for idx, item in enumerate(dataset_iter):
                # 检查是否达到最大样本数
                if max_samples and processed_count >= max_samples:
                    break
                
                # 从item创建task
                task = benchmark.get_task_from_item(item, idx)
                if task is None:
                    continue
                
                batch_tasks.append(task)
                batch_indices.append(idx)
                
                # 达到batch_size时处理一批
                if len(batch_tasks) >= batch_size:
                    batch_results = self._process_task_batch(benchmark, batch_tasks)
                    results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    if verbose:
                        pbar.update(len(batch_results))
                    
                    batch_tasks = []
                    batch_indices = []
            
            # 处理剩余的task
            if batch_tasks:
                batch_results = self._process_task_batch(benchmark, batch_tasks)
                results.extend(batch_results)
                processed_count += len(batch_results)
                if verbose:
                    pbar.update(len(batch_results))
        
        finally:
            if verbose:
                pbar.close()
        
        return results
    
    def _process_tasks(self, benchmark: BaseBenchmark, tasks_iter, batch_size: int) -> List[BenchmarkResult]:
        """处理任务列表（支持批量）"""
        if batch_size > 1:
            # 批量处理
            tasks_list = list(tasks_iter)
            results = []
            for i in range(0, len(tasks_list), batch_size):
                batch = tasks_list[i:i+batch_size]
                batch_results = self._process_task_batch(benchmark, batch)
                results.extend(batch_results)
            return results
        else:
            # 逐个处理
            return self._process_task_batch(benchmark, tasks_iter)
    
    def _process_task_batch(self, benchmark: BaseBenchmark, tasks) -> List[BenchmarkResult]:
        """处理一批任务"""
        results = []
        
        for task in tasks:
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
                    task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
                    question=task.question if hasattr(task, 'question') else "",
                    ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
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

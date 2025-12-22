# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # 测试执行器：执行benchmark测试
# # """

# # import time
# # from typing import List, Dict, Any, Optional

# # try:
# #     from tqdm import tqdm
# #     HAS_TQDM = True
# # except ImportError:
# #     HAS_TQDM = False
# #     # 简单的进度条替代
# #     def tqdm(iterable, desc=""):
# #         print(f"{desc}...")
# #         return iterable

# # import sys
# # from pathlib import Path

# # # 添加项目根目录到路径
# # current_dir = Path(__file__).parent.parent
# # sys.path.insert(0, str(current_dir))

# # from .model_adapter import BaseModelAdapter
# # from ..benchmarks.base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


# # class TestExecutor:
# #     """测试执行器"""
    
# #     def __init__(self, model_adapter: BaseModelAdapter):
# #         self.model_adapter = model_adapter
# #         self.model_info = model_adapter.get_model_info()
    
# #     def run_benchmark(self, 
# #                      benchmark: BaseBenchmark,
# #                      max_samples: Optional[int] = None,
# #                      verbose: bool = True) -> List[BenchmarkResult]:
# #         """
# #         运行单个benchmark测试
        
# #         Args:
# #             benchmark: Benchmark实例
# #             max_samples: 最大测试样本数（None表示全部）
# #             verbose: 是否显示进度
        
# #         Returns:
# #             测试结果列表
# #         """
# #         tasks = benchmark.get_tasks()
# #         if max_samples:
# #             tasks = tasks[:max_samples]
        
# #         results = []
        
# #         if verbose:
# #             tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}")
# #         else:
# #             tasks_iter = tasks
        
# #         for task in tasks_iter:
# #             try:
# #                 # 构建提示
# #                 prompt = self._build_prompt(task)
                
# #                 # 调用模型
# #                 model_response = self.model_adapter.generate(
# #                     prompt=prompt,
# #                     images=task.images
# #                 )
                
# #                 # 评估答案
# #                 result = benchmark.evaluate_answer(
# #                     model_answer=model_response.get("text", ""),
# #                     ground_truth=task.ground_truth,
# #                     task=task
# #                 )
                
# #                 results.append(result)
                
# #             except Exception as e:
# #                 # 记录错误
# #                 result = BenchmarkResult(
# #                     task_id=task.task_id,
# #                     question=task.question,
# #                     ground_truth=task.ground_truth,
# #                     model_answer="",
# #                     is_correct=False,
# #                     score=0.0,
# #                     metadata={"error": str(e)}
# #                 )
# #                 results.append(result)
        
# #         return results
    
# #     def _build_prompt(self, task: BenchmarkTask) -> str:
# #         """构建提示"""
# #         # 基本提示格式
# #         prompt = task.question
# #         return prompt







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
    
#     def __init__(self, model_adapter: BaseModelAdapter, verbose: bool = False):
#         self.model_adapter = model_adapter
#         self.model_info = model_adapter.get_model_info()
#         self._verbose = verbose  # 详细输出标志
    
#     def run_benchmark(self, 
#                      benchmark: BaseBenchmark,
#                      max_samples: Optional[int] = None,
#                      batch_size: int = 1,
#                      verbose: bool = True) -> List[BenchmarkResult]:
#         """
#         运行单个benchmark测试（支持流式和批量处理）
        
#         Args:
#             benchmark: Benchmark实例
#             max_samples: 最大测试样本数（None表示全部）
#             batch_size: 批处理大小（1表示逐个处理，>1表示批量处理）
#             verbose: 是否显示进度
        
#         Returns:
#             测试结果列表
#         """
#         # 更新详细输出标志
#         self._verbose = verbose
        
#         if verbose:
#             print(f"\n  🚀 开始测试 Benchmark: {benchmark.name}")
#             print(f"     模式: {'流式' if hasattr(benchmark, '_use_streaming') and benchmark._use_streaming else '批量'}")
#             print(f"     批大小: {batch_size}, 最大样本: {max_samples or '无限制'}")
        
#         # 检查是否支持流式加载
#         use_streaming = hasattr(benchmark, 'get_dataset_iterator') and hasattr(benchmark, 'get_task_from_item')
        
#         if use_streaming and hasattr(benchmark, '_use_streaming') and benchmark._use_streaming:
#             # 流式处理模式
#             results = self._run_benchmark_streaming(
#                 benchmark, max_samples, batch_size, verbose
#             )
#         else:
#             # 普通模式（一次性加载所有任务）
#             if verbose:
#                 print(f"  📦 加载所有任务...")
            
#             tasks = benchmark.get_tasks()
#             if max_samples:
#                 tasks = tasks[:max_samples]
            
#             if verbose:
#                 print(f"  ✓ 加载了 {len(tasks)} 个任务")
            
#             if verbose:
#                 tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}", unit="task")
#             else:
#                 tasks_iter = tasks
            
#             results = self._process_tasks(benchmark, tasks_iter, batch_size)
        
#         if verbose:
#             correct_count = sum(1 for r in results if r.is_correct)
#             error_count = sum(1 for r in results if r.metadata and 'error' in r.metadata)
#             print(f"  ✅ 测试完成: {correct_count}/{len(results)} 正确, {error_count} 错误")
        
#         return results
    
#     def _run_benchmark_streaming(self,
#                                 benchmark: BaseBenchmark,
#                                 max_samples: Optional[int],
#                                 batch_size: int,
#                                 verbose: bool) -> List[BenchmarkResult]:
#         """流式处理benchmark"""
#         results = []
        
#         if verbose:
#             print(f"  📊 开始流式处理，batch_size={batch_size}, max_samples={max_samples or 'all'}")
        
#         try:
#             dataset_iter = benchmark.get_dataset_iterator()
#         except Exception as e:
#             print(f"  ✗ 获取数据集迭代器失败: {e}")
#             import traceback
#             traceback.print_exc()
#             return results
        
#         processed_count = 0
#         skipped_count = 0
#         error_count = 0
        
#         # 创建进度条
#         if verbose:
#             if max_samples:
#                 pbar = tqdm(total=max_samples, desc=f"Testing {benchmark.name} (streaming)", unit="task")
#             else:
#                 pbar = tqdm(desc=f"Testing {benchmark.name} (streaming)", unit="task")
        
#         try:
#             batch_tasks = []
#             batch_indices = []
            
#             idx = 0
#             while True:
#                 try:
#                     # 检查是否达到最大样本数
#                     if max_samples and processed_count >= max_samples:
#                         if verbose:
#                             print(f"\n  ✓ 已达到最大样本数限制 ({max_samples})")
#                         break
                    
#                     # 尝试获取下一个item
#                     try:
#                         item = next(dataset_iter)
#                     except StopIteration:
#                         if verbose:
#                             print(f"\n  ✓ 数据集迭代完成 (共处理 {idx} 个items)")
#                         break
#                     except Exception as e:
#                         print(f"  ✗ 获取数据项失败 (idx={idx}): {e}")
#                         error_count += 1
#                         if verbose and error_count <= 10:
#                             import traceback
#                             traceback.print_exc()
#                         idx += 1
#                         continue
                    
#                     try:
#                         # 从item创建task
#                         task = benchmark.get_task_from_item(item, idx)
#                         if task is None:
#                             skipped_count += 1
#                             if verbose and skipped_count <= 5:
#                                 print(f"  ⚠️  跳过无效item (idx={idx})")
#                             idx += 1
#                             continue
                        
#                         batch_tasks.append(task)
#                         batch_indices.append(idx)
                        
#                         # 达到batch_size时处理一批
#                         if len(batch_tasks) >= batch_size:
#                             if verbose:
#                                 print(f"  🔄 处理批次 (size={len(batch_tasks)}, items {batch_indices[0]}-{batch_indices[-1]})")
                            
#                             try:
#                                 batch_results = self._process_task_batch(benchmark, batch_tasks)
#                                 results.extend(batch_results)
#                                 processed_count += len(batch_results)
                                
#                                 # 统计批次中的正确/错误数量
#                                 batch_correct = sum(1 for r in batch_results if r.is_correct)
#                                 batch_errors = sum(1 for r in batch_results if r.metadata and 'error' in r.metadata)
#                                 error_count += batch_errors
                                
#                                 if verbose:
#                                     pbar.update(len(batch_results))
#                                     print(f"    批次结果: {batch_correct}/{len(batch_results)} 正确, {batch_errors} 错误")
                            
#                             except Exception as e:
#                                 error_count += len(batch_tasks)
#                                 print(f"  ✗ 批次处理失败 (items {batch_indices[0]}-{batch_indices[-1]}): {e}")
#                                 if verbose:
#                                     import traceback
#                                     traceback.print_exc()
#                                 # 为失败的批次创建错误结果
#                                 for task in batch_tasks:
#                                     error_result = BenchmarkResult(
#                                         task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
#                                         question=task.question if hasattr(task, 'question') else "",
#                                         ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
#                                         model_answer="",
#                                         is_correct=False,
#                                         score=0.0,
#                                         metadata={"error": str(e)}
#                                     )
#                                     results.append(error_result)
                            
#                             batch_tasks = []
#                             batch_indices = []
                    
#                     except Exception as e:
#                         skipped_count += 1
#                         error_count += 1
#                         print(f"  ✗ 处理item失败 (idx={idx}): {e}")
#                         if verbose and error_count <= 10:
#                             import traceback
#                             traceback.print_exc()
                    
#                     idx += 1
                
#                 except KeyboardInterrupt:
#                     raise
                
#                 except StopIteration:
#                     break
                
#                 except Exception as e:
#                     print(f"  ✗ 迭代过程中发生错误 (idx={idx}): {e}")
#                     if verbose:
#                         import traceback
#                         traceback.print_exc()
#                     idx += 1
#                     continue
            
#             # 处理剩余的task
#             if batch_tasks:
#                 if verbose:
#                     print(f"  🔄 处理最后一批 (size={len(batch_tasks)}, items {batch_indices[0]}-{batch_indices[-1]})")
                
#                 try:
#                     batch_results = self._process_task_batch(benchmark, batch_tasks)
#                     results.extend(batch_results)
#                     processed_count += len(batch_results)
                    
#                     batch_correct = sum(1 for r in batch_results if r.is_correct)
#                     batch_errors = sum(1 for r in batch_results if r.metadata and 'error' in r.metadata)
#                     error_count += batch_errors
                    
#                     if verbose:
#                         pbar.update(len(batch_results))
#                         print(f"    批次结果: {batch_correct}/{len(batch_results)} 正确, {batch_errors} 错误")
                
#                 except Exception as e:
#                     error_count += len(batch_tasks)
#                     print(f"  ✗ 最后一批处理失败: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     # 为失败的批次创建错误结果
#                     for task in batch_tasks:
#                         error_result = BenchmarkResult(
#                             task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
#                             question=task.question if hasattr(task, 'question') else "",
#                             ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
#                             model_answer="",
#                             is_correct=False,
#                             score=0.0,
#                             metadata={"error": str(e)}
#                         )
#                         results.append(error_result)
        
#         except KeyboardInterrupt:
#             if verbose:
#                 print(f"\n  ⚠️  用户中断测试")
#             raise
        
#         except Exception as e:
#             print(f"  ✗ 流式处理发生严重错误: {e}")
#             import traceback
#             traceback.print_exc()
        
#         finally:
#             if verbose:
#                 pbar.close()
#                 print(f"  📊 流式处理完成:")
#                 print(f"    总处理数: {processed_count}")
#                 print(f"    跳过数: {skipped_count}")
#                 print(f"    错误数: {error_count}")
#                 print(f"    成功结果: {len(results)}")
        
#         return results
    
#     def _process_tasks(self, benchmark: BaseBenchmark, tasks_iter, batch_size: int) -> List[BenchmarkResult]:
#         """处理任务列表（支持批量）"""
#         if batch_size > 1:
#             # 批量处理
#             tasks_list = list(tasks_iter)
#             results = []
#             for i in range(0, len(tasks_list), batch_size):
#                 batch = tasks_list[i:i+batch_size]
#                 batch_results = self._process_task_batch(benchmark, batch)
#                 results.extend(batch_results)
#             return results
#         else:
#             # 逐个处理
#             return self._process_task_batch(benchmark, tasks_iter)
    
#     def _process_task_batch(self, benchmark: BaseBenchmark, tasks) -> List[BenchmarkResult]:
#         """处理一批任务"""
#         results = []
        
#         for i, task in enumerate(tasks):
#             try:
#                 # 构建提示
#                 prompt = self._build_prompt(task)
                
#                 # 调用模型
#                 if hasattr(self, '_verbose') and self._verbose:
#                     print(f"      [{i+1}/{len(tasks)}] 处理任务: {task.task_id[:50]}...")
                
#                 model_response = self.model_adapter.generate(
#                     prompt=prompt,
#                     images=task.images
#                 )
                
#                 model_answer = model_response.get("text", "")
                
#                 # 评估答案
#                 result = benchmark.evaluate_answer(
#                     model_answer=model_answer,
#                     ground_truth=task.ground_truth,
#                     task=task
#                 )
                
#                 results.append(result)
                
#                 if hasattr(self, '_verbose') and self._verbose:
#                     status = "✓" if result.is_correct else "✗"
#                     print(f"        {status} 答案: {model_answer[:50]}... | GT: {task.ground_truth}")
                
#             except KeyboardInterrupt:
#                 raise
            
#             except Exception as e:
#                 # 记录错误，但继续处理其他任务
#                 error_msg = str(e)
#                 if hasattr(self, '_verbose') and self._verbose:
#                     print(f"        ✗ 任务处理错误: {error_msg[:100]}")
                
#                 result = BenchmarkResult(
#                     task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
#                     question=task.question if hasattr(task, 'question') else "",
#                     ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
#                     model_answer="",
#                     is_correct=False,
#                     score=0.0,
#                     metadata={"error": error_msg}
#                 )
#                 results.append(result)
        
#         return results
    
#     def _build_prompt(self, task: BenchmarkTask) -> str:
#         """构建提示"""
#         # 基本提示格式
#         prompt = task.question
#         return prompt




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
    
    def __init__(self, model_adapter: BaseModelAdapter, verbose: bool = False):
        self.model_adapter = model_adapter
        self.model_info = model_adapter.get_model_info()
        self._verbose = verbose  # 详细输出标志
    
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
        # 更新详细输出标志
        self._verbose = verbose
        
        if verbose:
            print(f"\n  🚀 开始测试 Benchmark: {benchmark.name}")
            print(f"     模式: {'流式' if hasattr(benchmark, '_use_streaming') and benchmark._use_streaming else '批量'}")
            print(f"     批大小: {batch_size}, 最大样本: {max_samples or '无限制'}")
        
        # 检查是否支持流式加载
        use_streaming = hasattr(benchmark, 'get_dataset_iterator') and hasattr(benchmark, 'get_task_from_item')
        
        if use_streaming and hasattr(benchmark, '_use_streaming') and benchmark._use_streaming:
            # 流式处理模式
            results = self._run_benchmark_streaming(
                benchmark, max_samples, batch_size, verbose
            )
        else:
            # 普通模式（一次性加载所有任务）
            if verbose:
                print(f"  📦 加载所有任务...")
            
            tasks = benchmark.get_tasks()
            if max_samples:
                tasks = tasks[:max_samples]
            
            if verbose:
                print(f"  ✓ 加载了 {len(tasks)} 个任务")
            
            if verbose:
                tasks_iter = tqdm(tasks, desc=f"Testing {benchmark.name}", unit="task")
            else:
                tasks_iter = tasks
            
            results = self._process_tasks(benchmark, tasks_iter, batch_size)
        
        if verbose:
            correct_count = sum(1 for r in results if r.is_correct)
            error_count = sum(1 for r in results if r.metadata and 'error' in r.metadata)
            print(f"  ✅ 测试完成: {correct_count}/{len(results)} 正确, {error_count} 错误")
        
        return results
    
    def _run_benchmark_streaming(self,
                                benchmark: BaseBenchmark,
                                max_samples: Optional[int],
                                batch_size: int,
                                verbose: bool) -> List[BenchmarkResult]:
        """流式处理benchmark"""
        results = []
        
        if verbose:
            print(f"  📊 开始流式处理，batch_size={batch_size}, max_samples={max_samples or 'all'}")
        
        try:
            dataset_iter = benchmark.get_dataset_iterator()
        except Exception as e:
            print(f"  ✗ 获取数据集迭代器失败: {e}")
            import traceback
            traceback.print_exc()
            return results
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # 创建进度条
        if verbose:
            if max_samples:
                pbar = tqdm(total=max_samples, desc=f"Testing {benchmark.name} (streaming)", unit="task")
            else:
                pbar = tqdm(desc=f"Testing {benchmark.name} (streaming)", unit="task")
        
        try:
            batch_tasks = []
            batch_indices = []
            
            idx = 0
            while True:
                try:
                    # 检查是否达到最大样本数
                    if max_samples and processed_count >= max_samples:
                        if verbose:
                            print(f"\n  ✓ 已达到最大样本数限制 ({max_samples})")
                        break
                    
                    # 尝试获取下一个item
                    try:
                        item = next(dataset_iter)
                    except StopIteration:
                        if verbose:
                            print(f"\n  ✓ 数据集迭代完成 (共处理 {idx} 个items)")
                        break
                    except Exception as e:
                        print(f"  ✗ 获取数据项失败 (idx={idx}): {e}")
                        error_count += 1
                        if verbose and error_count <= 10:
                            import traceback
                            traceback.print_exc()
                        idx += 1
                        continue
                    
                    try:
                        # 从item创建task
                        task = benchmark.get_task_from_item(item, idx)
                        if task is None:
                            skipped_count += 1
                            if verbose and skipped_count <= 5:
                                print(f"  ⚠️  跳过无效item (idx={idx})")
                            idx += 1
                            continue
                        
                        batch_tasks.append(task)
                        batch_indices.append(idx)
                        
                        # 达到batch_size时处理一批
                        if len(batch_tasks) >= batch_size:
                            if verbose:
                                print(f"  🔄 处理批次 (size={len(batch_tasks)}, items {batch_indices[0]}-{batch_indices[-1]})")
                            
                            try:
                                batch_results = self._process_task_batch(benchmark, batch_tasks)
                                results.extend(batch_results)
                                processed_count += len(batch_results)
                                
                                # 统计批次中的正确/错误数量
                                batch_correct = sum(1 for r in batch_results if r.is_correct)
                                batch_errors = sum(1 for r in batch_results if r.metadata and 'error' in r.metadata)
                                error_count += batch_errors
                                
                                if verbose:
                                    pbar.update(len(batch_results))
                                    print(f"    批次结果: {batch_correct}/{len(batch_results)} 正确, {batch_errors} 错误")
                            
                            except Exception as e:
                                error_count += len(batch_tasks)
                                print(f"  ✗ 批次处理失败 (items {batch_indices[0]}-{batch_indices[-1]}): {e}")
                                if verbose:
                                    import traceback
                                    traceback.print_exc()
                                # 为失败的批次创建错误结果
                                for task in batch_tasks:
                                    error_result = BenchmarkResult(
                                        task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
                                        question=task.question if hasattr(task, 'question') else "",
                                        ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
                                        model_answer="",
                                        is_correct=False,
                                        score=0.0,
                                        metadata={"error": str(e)}
                                    )
                                    results.append(error_result)
                            
                            batch_tasks = []
                            batch_indices = []
                    
                    except Exception as e:
                        skipped_count += 1
                        error_count += 1
                        print(f"  ✗ 处理item失败 (idx={idx}): {e}")
                        if verbose and error_count <= 10:
                            import traceback
                            traceback.print_exc()
                    
                    idx += 1
                
                except KeyboardInterrupt:
                    raise
                
                except StopIteration:
                    break
                
                except Exception as e:
                    print(f"  ✗ 迭代过程中发生错误 (idx={idx}): {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    idx += 1
                    continue
            
            # 处理剩余的task
            if batch_tasks:
                if verbose:
                    print(f"  🔄 处理最后一批 (size={len(batch_tasks)}, items {batch_indices[0]}-{batch_indices[-1]})")
                
                try:
                    batch_results = self._process_task_batch(benchmark, batch_tasks)
                    results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    batch_correct = sum(1 for r in batch_results if r.is_correct)
                    batch_errors = sum(1 for r in batch_results if r.metadata and 'error' in r.metadata)
                    error_count += batch_errors
                    
                    if verbose:
                        pbar.update(len(batch_results))
                        print(f"    批次结果: {batch_correct}/{len(batch_results)} 正确, {batch_errors} 错误")
                
                except Exception as e:
                    error_count += len(batch_tasks)
                    print(f"  ✗ 最后一批处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 为失败的批次创建错误结果
                    for task in batch_tasks:
                        error_result = BenchmarkResult(
                            task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
                            question=task.question if hasattr(task, 'question') else "",
                            ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
                            model_answer="",
                            is_correct=False,
                            score=0.0,
                            metadata={"error": str(e)}
                        )
                        results.append(error_result)
        
        except KeyboardInterrupt:
            if verbose:
                print(f"\n  ⚠️  用户中断测试")
            raise
        
        except Exception as e:
            print(f"  ✗ 流式处理发生严重错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if verbose:
                pbar.close()
                print(f"  📊 流式处理完成:")
                print(f"    总处理数: {processed_count}")
                print(f"    跳过数: {skipped_count}")
                print(f"    错误数: {error_count}")
                print(f"    成功结果: {len(results)}")
        
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
        
        for i, task in enumerate(tasks):
            try:
                # 构建提示
                prompt = self._build_prompt(task)
                
                # 调用模型
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"      [{i+1}/{len(tasks)}] 处理任务: {task.task_id[:50]}...")
                    print(f"        📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"        🖼️  图像数量: {len(task.images) if task.images else 0}")
                
                model_response = self.model_adapter.generate(
                    prompt=prompt,
                    images=task.images
                )

                # 输出模型返回值的调试信息
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"        🔍 模型返回值类型: {type(model_response)}")
                    print(f"        🔍 模型返回值的键: {list(model_response.keys()) if isinstance(model_response, dict) else 'N/A'}")
                    # 输出完整的返回值（如果是字典，格式化输出）
                    if isinstance(model_response, dict):
                        print(f"        🔍 模型返回值内容:")
                        for key, value in model_response.items():
                            if key == "raw" and isinstance(value, dict):
                                # raw字段可能很大，只显示类型和部分信息
                                print(f"          - {key}: {type(value).__name__} (隐藏详细内容)")
                            else:
                                # 限制字符串长度，避免输出过长
                                value_str = str(value)
                                if len(value_str) > 200:
                                    value_str = value_str[:200] + "... (已截断)"
                                print(f"          - {key}: {value_str}")
                    else:
                        print(f"        🔍 模型返回值: {model_response}")
                
                model_answer = model_response.get("text", "")
                
                # 评估答案
                result = benchmark.evaluate_answer(
                    model_answer=model_answer,
                    ground_truth=task.ground_truth,
                    task=task
                )
                
                results.append(result)
                
                if hasattr(self, '_verbose') and self._verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(f"        {status} 答案: {model_answer[:50]}... | GT: {task.ground_truth}")
                
            except KeyboardInterrupt:
                raise
            
            except Exception as e:
                # 记录错误，但继续处理其他任务
                error_msg = str(e)
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"        ✗ 任务处理错误: {error_msg[:100]}")
                
                result = BenchmarkResult(
                    task_id=task.task_id if hasattr(task, 'task_id') else f"error_{len(results)}",
                    question=task.question if hasattr(task, 'question') else "",
                    ground_truth=task.ground_truth if hasattr(task, 'ground_truth') else None,
                    model_answer="",
                    is_correct=False,
                    score=0.0,
                    metadata={"error": error_msg}
                )
                results.append(result)
        
        return results
    
    def _build_prompt(self, task: BenchmarkTask) -> str:
        """
        构建提示
        
        对于不同的模型类型，可能需要不同的prompt格式：
        - BLIP VQA模型：通常直接使用问题文本即可
        - 其他VQA模型：可能需要 "Question: ... Answer:" 格式
        """
        # 基本提示格式：直接使用问题文本
        prompt = task.question
        
        # 可以根据模型类型调整prompt格式
        # 例如对于某些模型，可能需要：
        # prompt = f"Question: {task.question} Answer:"
        # 或者
        # prompt = f"Q: {task.question}\nA:"
        
        return prompt

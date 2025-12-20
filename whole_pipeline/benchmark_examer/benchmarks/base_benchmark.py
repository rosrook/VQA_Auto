# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Benchmark基类
# """

# import abc
# from typing import Dict, List, Any, Optional
# from dataclasses import dataclass


# @dataclass
# class BenchmarkTask:
#     """Benchmark任务"""
#     task_id: str
#     question: str
#     images: List[str]  # 图像路径或base64
#     ground_truth: Any  # 正确答案
#     metadata: Dict[str, Any] = None  # 额外信息
    
#     def __post_init__(self):
#         if self.metadata is None:
#             self.metadata = {}


# @dataclass
# class BenchmarkResult:
#     """Benchmark测试结果"""
#     task_id: str
#     question: str
#     ground_truth: Any
#     model_answer: str
#     is_correct: bool
#     score: float  # 0-1之间的分数
#     metadata: Dict[str, Any] = None
    
#     def __post_init__(self):
#         if self.metadata is None:
#             self.metadata = {}


# class BaseBenchmark(abc.ABC):
#     """Benchmark基类"""
    
#     def __init__(self, name: str, data_path: str, **kwargs):
#         self.name = name
#         self.data_path = data_path
#         self.kwargs = kwargs
#         self.tasks: List[BenchmarkTask] = []
#         self._load_data()
    
#     @abc.abstractmethod
#     def _load_data(self):
#         """加载benchmark数据"""
#         raise NotImplementedError
    
#     @abc.abstractmethod
#     def evaluate_answer(self, 
#                        model_answer: str, 
#                        ground_truth: Any,
#                        task: BenchmarkTask) -> BenchmarkResult:
#         """
#         评估模型答案
        
#         Args:
#             model_answer: 模型的回答
#             ground_truth: 正确答案
#             task: 任务信息
        
#         Returns:
#             BenchmarkResult
#         """
#         raise NotImplementedError
    
#     def get_tasks(self) -> List[BenchmarkTask]:
#         """获取所有任务"""
#         return self.tasks
    
#     def get_info(self) -> Dict[str, Any]:
#         """获取benchmark信息"""
#         return {
#             "name": self.name,
#             "data_path": self.data_path,
#             "num_tasks": len(self.tasks),
#             "description": getattr(self, "description", "")
#         }


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark基类
"""

import abc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkTask:
    """Benchmark任务"""
    task_id: str
    question: str
    images: List[str]  # 图像路径或base64
    ground_truth: Any  # 正确答案
    metadata: Dict[str, Any] = None  # 额外信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """Benchmark测试结果"""
    task_id: str
    question: str
    ground_truth: Any
    model_answer: str
    is_correct: bool
    score: float  # 0-1之间的分数
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmark(abc.ABC):
    """Benchmark基类"""
    
    def __init__(self, name: str, data_path: str, load_data_on_init: bool = True, **kwargs):
        self.name = name
        self.data_path = data_path
        self.kwargs = kwargs
        self.tasks: List[BenchmarkTask] = []
        
        # 允许子类控制是否在初始化时加载数据
        if load_data_on_init:
            self._load_data()
    
    @abc.abstractmethod
    def _load_data(self):
        """加载benchmark数据"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate_answer(self, 
                       model_answer: str, 
                       ground_truth: Any,
                       task: BenchmarkTask) -> BenchmarkResult:
        """
        评估模型答案
        
        Args:
            model_answer: 模型的回答
            ground_truth: 正确答案
            task: 任务信息
        
        Returns:
            BenchmarkResult
        """
        raise NotImplementedError
    
    def get_tasks(self) -> List[BenchmarkTask]:
        """获取所有任务"""
        return self.tasks
    
    def get_info(self) -> Dict[str, Any]:
        """获取benchmark信息"""
        return {
            "name": self.name,
            "data_path": self.data_path,
            "num_tasks": len(self.tasks),
            "description": getattr(self, "description", "")
        }

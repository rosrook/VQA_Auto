"""
Benchmark测试系统：自动测试模型在不同benchmark上的表现
"""

from .core.model_adapter import BaseModelAdapter, ModelAdapterFactory
from .core.benchmark_manager import BenchmarkManager
from .core.test_executor import TestExecutor
from .core.result_analyzer import ResultAnalyzer

__all__ = [
    'BaseModelAdapter',
    'ModelAdapterFactory',
    'BenchmarkManager',
    'TestExecutor',
    'ResultAnalyzer'
]

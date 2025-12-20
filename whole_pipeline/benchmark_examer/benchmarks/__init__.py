"""
Benchmark定义模块
"""

from .base_benchmark import BaseBenchmark
from .huggingface_benchmark import HuggingFaceBenchmark

__all__ = ['BaseBenchmark', 'HuggingFaceBenchmark']

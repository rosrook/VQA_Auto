#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark管理器：管理所有benchmark，根据需求自动选择合适的benchmark
"""

import json
import os
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from ..benchmarks.base_benchmark import BaseBenchmark, BenchmarkTask
from ..benchmarks.huggingface_benchmark import HuggingFaceBenchmark


class BenchmarkManager:
    """Benchmark管理器"""
    
    def __init__(self, 
                 benchmarks_dir: str = None,
                 taxonomy_file: str = None,
                 model_type: str = None):
        """
        Args:
            benchmarks_dir: benchmark目录（已废弃，保留用于兼容）
            taxonomy_file: benchmark分类文件路径
            model_type: 模型类型（用于筛选benchmark）
        """
        self.benchmarks_dir = benchmarks_dir
        self.model_type = model_type
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        self.benchmark_catalog: Dict[str, Dict] = {}
        
        # 加载benchmark分类文件
        if taxonomy_file:
            self._load_taxonomy_file(taxonomy_file)
        else:
            # 默认路径
            default_taxonomy = Path(__file__).parent.parent / "benchmarks" / "available_benchmarks_with_internal_taxonomy.json"
            if default_taxonomy.exists():
                self._load_taxonomy_file(str(default_taxonomy))
        
        self._load_benchmarks()
    
    def _load_taxonomy_file(self, file_path: str):
        """加载benchmark分类文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                catalog_data = json.load(f)
                self.benchmark_catalog = {
                    bench["name"]: bench 
                    for bench in catalog_data.get("benchmarks", [])
                }
        except Exception as e:
            print(f"警告: 加载benchmark分类文件失败: {e}")
            self.benchmark_catalog = {}
    
    def _load_benchmarks(self):
        """从HuggingFace加载benchmark"""
        if not self.benchmark_catalog:
            return
        
        # 根据model_type筛选要加载的benchmark
        benchmarks_to_load = self._filter_benchmarks_by_model_type()
        
        for bench_name, bench_info in benchmarks_to_load.items():
            if bench_info.get("source") == "huggingface":
                try:
                    benchmark = HuggingFaceBenchmark(
                        name=bench_name,
                        hf_id=bench_info["hf_id"],
                        config=bench_info.get("config", "default"),
                        split=bench_info.get("default_split", "validation"),
                        benchmark_info=bench_info
                    )
                    self.benchmarks[bench_name] = benchmark
                    print(f"✓ 加载benchmark: {bench_name} (from {bench_info['hf_id']})")
                except Exception as e:
                    print(f"✗ 加载benchmark失败 {bench_name}: {e}")
    
    def _filter_benchmarks_by_model_type(self) -> Dict[str, Dict]:
        """
        根据model_type筛选benchmark
        
        如果model_type为None，返回所有benchmark
        """
        if self.model_type is None:
            return self.benchmark_catalog
        
        # 如果JSON中有model_type字段，可以根据它筛选
        # 否则返回所有（用户可以在requirements中进一步筛选）
        return self.benchmark_catalog
    
    def register_benchmark(self, name: str, benchmark: BaseBenchmark):
        """注册benchmark"""
        self.benchmarks[name] = benchmark
    
    def get_benchmark(self, name: str) -> Optional[BaseBenchmark]:
        """获取指定的benchmark"""
        return self.benchmarks.get(name)
    
    def list_benchmarks(self) -> List[str]:
        """列出所有可用的benchmark"""
        return list(self.benchmarks.keys())
    
    def select_benchmarks(self, 
                         requirements: Dict[str, Any] = None,
                         benchmark_names: List[str] = None) -> List[BaseBenchmark]:
        """
        根据需求选择合适的benchmark
        
        Args:
            requirements: 需求字典，例如：
                {
                    "task_types": ["vqa", "counting"],
                    "difficulty": "medium",
                    "min_samples": 100
                }
            benchmark_names: 指定要使用的benchmark名称列表（优先级高于requirements）
        
        Returns:
            选中的benchmark列表
        """
        # 如果指定了benchmark名称，直接返回
        if benchmark_names:
            selected = []
            for name in benchmark_names:
                if name in self.benchmarks:
                    selected.append(self.benchmarks[name])
                else:
                    print(f"警告: benchmark '{name}' 不存在")
            return selected
        
        if requirements is None:
            # 如果没有指定需求，返回所有benchmark
            return list(self.benchmarks.values())
        
        selected = []
        for name, benchmark in self.benchmarks.items():
            info = benchmark.get_info()
            
            # 根据需求筛选
            if "task_types" in requirements:
                benchmark_types = info.get("task_types", [])
                required_types = requirements["task_types"]
                if not any(t in benchmark_types for t in required_types):
                    continue
            
            if "min_samples" in requirements:
                if info.get("num_tasks", 0) < requirements["min_samples"]:
                    continue
            
            # 可以根据taxonomy字段筛选
            if "taxonomy_fields" in requirements:
                native_fields = info.get("native_taxonomy_fields", [])
                required_fields = requirements["taxonomy_fields"]
                if not any(f in native_fields for f in required_fields):
                    continue
            
            selected.append(benchmark)
        
        return selected
    
    def list_available_benchmarks(self) -> List[Dict[str, Any]]:
        """列出所有可用的benchmark（从catalog）"""
        return list(self.benchmark_catalog.values())

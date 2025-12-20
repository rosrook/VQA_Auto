#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例VQA Benchmark实现
"""

import json
from typing import List, Dict, Any
from .base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


class ExampleVQABenchmark(BaseBenchmark):
    """示例VQA Benchmark"""
    
    def __init__(self, name: str = "example_vqa", data_path: str = None, **kwargs):
        self.description = "示例VQA Benchmark，用于演示"
        super().__init__(name, data_path or "data/example_vqa.json", **kwargs)
    
    def _load_data(self):
        """加载benchmark数据"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 假设数据格式为列表
            if isinstance(data, list):
                tasks_data = data
            elif isinstance(data, dict) and "data" in data:
                tasks_data = data["data"]
            else:
                tasks_data = [data]
            
            self.tasks = []
            for item in tasks_data:
                task = BenchmarkTask(
                    task_id=item.get("id", f"task_{len(self.tasks)}"),
                    question=item.get("question", ""),
                    images=item.get("images", []),  # 图像路径列表
                    ground_truth=item.get("answer", item.get("ground_truth", "")),
                    metadata=item.get("metadata", {})
                )
                self.tasks.append(task)
                
        except FileNotFoundError:
            print(f"警告: 数据文件不存在 {self.data_path}，使用空数据")
            self.tasks = []
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.tasks = []
    
    def evaluate_answer(self, 
                       model_answer: str, 
                       ground_truth: Any,
                       task: BenchmarkTask) -> BenchmarkResult:
        """
        评估模型答案
        
        简单实现：字符串完全匹配
        可以根据需要实现更复杂的评估逻辑（模糊匹配、语义相似度等）
        """
        model_answer_clean = str(model_answer).strip().lower()
        ground_truth_clean = str(ground_truth).strip().lower()
        
        # 简单匹配
        is_correct = model_answer_clean == ground_truth_clean
        
        # 部分匹配（如果答案包含关键词）
        if not is_correct:
            # 可以扩展更复杂的匹配逻辑
            if ground_truth_clean in model_answer_clean or model_answer_clean in ground_truth_clean:
                score = 0.5
            else:
                score = 0.0
        else:
            score = 1.0
        
        return BenchmarkResult(
            task_id=task.task_id,
            question=task.question,
            ground_truth=ground_truth,
            model_answer=model_answer,
            is_correct=is_correct,
            score=score,
            metadata={"evaluation_method": "exact_match"}
        )

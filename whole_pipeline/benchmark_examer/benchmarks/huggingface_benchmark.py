# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# HuggingFace Benchmark：从HuggingFace Hub加载benchmark数据
# """

# from typing import List, Dict, Any, Optional
# from .base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


# class HuggingFaceBenchmark(BaseBenchmark):
#     """从HuggingFace Hub加载的Benchmark"""
    
#     def __init__(self, 
#                  name: str,
#                  hf_id: str,
#                  config: str = "default",
#                  split: str = None,
#                  **kwargs):
#         """
#         Args:
#             name: Benchmark名称
#             hf_id: HuggingFace数据集ID（如 "gqa", "clevr"）
#             config: 数据集配置名（如 "balanced", "default"）
#             split: 数据集split（如 "validation", "test"），如果为None则使用默认split
#         """
#         self.hf_id = hf_id
#         self.config = config
#         self.split = split
#         self.benchmark_info = kwargs.get("benchmark_info", {})
        
#         # 如果没有指定split，使用默认split
#         if self.split is None:
#             self.split = self.benchmark_info.get("default_split", "validation")
        
#         # data_path参数在这里不使用，但需要传递以满足基类要求
#         super().__init__(name, f"hf://{hf_id}/{config}/{self.split}", **kwargs)
#         self.description = f"HuggingFace Benchmark: {name} ({hf_id})"
    
#     def _load_data(self):
#         """从HuggingFace Hub加载数据"""
#         try:
#             from datasets import load_dataset
#         except ImportError:
#             raise ImportError(
#                 "需要安装datasets库: pip install datasets"
#             )
        
#         try:
#             # 加载数据集
#             if self.config == "default" or self.config is None:
#                 dataset = load_dataset(self.hf_id, split=self.split)
#             else:
#                 dataset = load_dataset(self.hf_id, name=self.config, split=self.split)
            
#             self.tasks = []
            
#             # 将数据集转换为BenchmarkTask列表
#             for idx, item in enumerate(dataset):
#                 task = self._convert_to_task(item, idx)
#                 if task:
#                     self.tasks.append(task)
        
#         except Exception as e:
#             raise RuntimeError(f"从HuggingFace加载数据失败 ({self.hf_id}): {e}")
    
#     def _convert_to_task(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkTask]:
#         """
#         将数据集item转换为BenchmarkTask
        
#         不同benchmark的数据格式可能不同，需要适配
#         """
#         task_id = item.get("id", item.get("question_id", f"task_{idx}"))
        
#         # 获取问题
#         question = item.get("question", item.get("text", ""))
#         if not question:
#             return None
        
#         # 获取图像
#         images = []
#         if "image" in item:
#             # PIL Image对象
#             images.append(item["image"])
#         elif "image_path" in item:
#             images.append(item["image_path"])
#         elif "image_file" in item:
#             images.append(item["image_file"])
#         elif "img" in item:
#             images.append(item["img"])
        
#         # 获取正确答案
#         ground_truth = item.get("answer", item.get("answers", None))
#         if ground_truth is None:
#             ground_truth = item.get("label", item.get("target", ""))
        
#         # 如果ground_truth是列表，取第一个
#         if isinstance(ground_truth, list) and len(ground_truth) > 0:
#             ground_truth = ground_truth[0]
        
#         # 提取metadata（包括taxonomy信息）
#         metadata = {}
#         taxonomy_fields = self.benchmark_info.get("native_taxonomy_fields", [])
#         for field in taxonomy_fields:
#             if field in item:
#                 metadata[field] = item[field]
        
#         # 保留其他有用字段
#         for key in ["question_type", "semantic", "program", "question_family", 
#                    "chart_type", "category", "capability", "task_group"]:
#             if key in item and key not in metadata:
#                 metadata[key] = item[key]
        
#         return BenchmarkTask(
#             task_id=str(task_id),
#             question=str(question),
#             images=images if images else [],
#             ground_truth=ground_truth,
#             metadata=metadata
#         )
    
#     def evaluate_answer(self, 
#                        model_answer: str, 
#                        ground_truth: Any,
#                        task: BenchmarkTask) -> BenchmarkResult:
#         """
#         评估模型答案
        
#         简单实现：字符串匹配
#         可以根据不同benchmark的特点实现更复杂的评估逻辑
#         """
#         model_answer_clean = str(model_answer).strip().lower()
#         ground_truth_clean = str(ground_truth).strip().lower()
        
#         # 精确匹配
#         is_correct = model_answer_clean == ground_truth_clean
        
#         # 部分匹配（如果答案包含关键词）
#         if not is_correct:
#             if ground_truth_clean in model_answer_clean or model_answer_clean in ground_truth_clean:
#                 score = 0.5
#             else:
#                 score = 0.0
#         else:
#             score = 1.0
        
#         # 可以在这里实现特定benchmark的评估逻辑
#         # 例如GQA可能需要特殊处理，CLEVR需要执行程序等
        
#         return BenchmarkResult(
#             task_id=task.task_id,
#             question=task.question,
#             ground_truth=ground_truth,
#             model_answer=model_answer,
#             is_correct=is_correct,
#             score=score,
#             metadata={"evaluation_method": "exact_match", "benchmark": self.name}
#         )
    
#     def get_info(self) -> Dict[str, Any]:
#         """获取benchmark信息"""
#         info = super().get_info()
#         info.update({
#             "hf_id": self.hf_id,
#             "config": self.config,
#             "split": self.split,
#             "source": "huggingface",
#             "native_taxonomy_fields": self.benchmark_info.get("native_taxonomy_fields", [])
#         })
#         return info





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Benchmark：从HuggingFace Hub加载benchmark数据
"""

from typing import List, Dict, Any, Optional
from .base_benchmark import BaseBenchmark, BenchmarkTask, BenchmarkResult


class HuggingFaceBenchmark(BaseBenchmark):
    """从HuggingFace Hub加载的Benchmark"""
    
    def __init__(self, 
                 name: str,
                 hf_id: str,
                 config: str = "default",
                 split: str = None,
                 **kwargs):
        """
        Args:
            name: Benchmark名称
            hf_id: HuggingFace数据集ID（完整ID，如 "lmms-lab/GQA"）
            config: 数据集配置名（如 "balanced", "default"）
            split: 数据集split（如 "validation", "test"），如果为None则使用默认split
            **kwargs: 其他参数，包括 benchmark_info
        """
        self.hf_id = hf_id
        self.config = config if config and config != "default" else None
        self.benchmark_info = kwargs.get("benchmark_info", {})
        
        # 如果没有指定split，使用benchmark_info中的默认split
        if split is None:
            split = self.benchmark_info.get("default_split", "validation")
        self.split = split
        
        # 获取可用的splits（用于验证）
        self.available_splits = self.benchmark_info.get("available_splits", [])
        
        # data_path参数在这里不使用，但需要传递以满足基类要求
        data_path = f"hf://{self.hf_id}"
        if self.config:
            data_path += f"/{self.config}"
        data_path += f"/{self.split}"
        
        super().__init__(name, data_path, **kwargs)
        self.description = f"HuggingFace Benchmark: {name} ({self.hf_id})"
    
    # def _load_data(self):
    #     """从HuggingFace Hub加载数据"""
    #     try:
    #         from datasets import load_dataset
    #     except ImportError:
    #         raise ImportError(
    #             "需要安装datasets库: pip install datasets"
    #         )
        
    #     try:
    #         print(f"🔄 正在加载数据集: {self.hf_id}")
    #         print(f"   配置: {self.config or 'default'}, Split: {self.split}")
            
    #         # 验证split是否可用
    #         if self.available_splits and self.split not in self.available_splits:
    #             print(f"⚠️  警告: split '{self.split}' 不在可用列表中 {self.available_splits}")
            
    #         # 加载数据集
    #         load_kwargs = {
    #             "split": self.split,
    #             "trust_remote_code": True
    #         }
            
    #         if self.config:
    #             load_kwargs["name"] = self.config
            
    #         dataset = load_dataset(self.hf_id, **load_kwargs)
            
    #         print(f"✓ 成功加载 {len(dataset)} 条数据")
            
    #         self.tasks = []
            
    #         # 将数据集转换为BenchmarkTask列表
    #         for idx, item in enumerate(dataset):
    #             task = self._convert_to_task(item, idx)
    #             if task:
    #                 self.tasks.append(task)
            
    #         print(f"✓ 转换完成，共 {len(self.tasks)} 个任务")
        
    #     except Exception as e:
    #         # 提供更详细的错误信息
    #         error_msg = f"从HuggingFace加载数据失败 ({self.hf_id}): {e}"
    #         print(f"✗ {error_msg}")
            
    #         # 提供可能的解决方案
    #         suggestions = []
    #         if "doesn't exist" in str(e) or "cannot be accessed" in str(e):
    #             suggestions.append(f"• 检查数据集ID是否正确: {self.hf_id}")
    #             suggestions.append(f"• 访问 https://huggingface.co/datasets/{self.hf_id} 确认数据集存在")
    #             if self.config:
    #                 suggestions.append(f"• 检查配置名称是否正确: {self.config}")
    #             suggestions.append(f"• 检查split是否正确: {self.split}")
    #             if self.available_splits:
    #                 suggestions.append(f"• 可用的splits: {self.available_splits}")
    #         elif "split" in str(e).lower():
    #             suggestions.append(f"• 当前使用的split: {self.split}")
    #             if self.available_splits:
    #                 suggestions.append(f"• 可用的splits: {self.available_splits}")
    #             suggestions.append(f"• 尝试使用其他split，或在config中更新available_splits")
            
    #         if suggestions:
    #             error_msg += "\n建议:\n" + "\n".join(suggestions)
            
    #         raise RuntimeError(error_msg)

    def _load_data(self):
        """从HuggingFace Hub加载数据"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "需要安装datasets库: pip install datasets"
            )
        
        try:
            print(f"🔄 正在加载数据集: {self.hf_id}")
            print(f"   配置: {self.config or 'default'}, Split: {self.split}")
            
            # 验证split是否可用
            if self.available_splits and self.split not in self.available_splits:
                print(f"⚠️  警告: split '{self.split}' 不在可用列表中 {self.available_splits}")
            
            # 加载数据集
            load_kwargs = {
                "split": self.split
            }
            
            if self.config:
                load_kwargs["name"] = self.config
            
            dataset = load_dataset(self.hf_id, **load_kwargs)
            
            print(f"✓ 成功加载 {len(dataset)} 条数据")
            
            self.tasks = []
            
            # 将数据集转换为BenchmarkTask列表
            for idx, item in enumerate(dataset):
                task = self._convert_to_task(item, idx)
                if task:
                    self.tasks.append(task)
            
            print(f"✓ 转换完成，共 {len(self.tasks)} 个任务")
        
        except Exception as e:
            # 提供更详细的错误信息
            error_msg = f"从HuggingFace加载数据失败 ({self.hf_id}): {e}"
            print(f"✗ {error_msg}")
            
            # 提供可能的解决方案
            suggestions = []
            if "doesn't exist" in str(e) or "cannot be accessed" in str(e):
                suggestions.append(f"• 检查数据集ID是否正确: {self.hf_id}")
                suggestions.append(f"• 访问 https://huggingface.co/datasets/{self.hf_id} 确认数据集存在")
                if self.config:
                    suggestions.append(f"• 检查配置名称是否正确: {self.config}")
                suggestions.append(f"• 检查split是否正确: {self.split}")
                if self.available_splits:
                    suggestions.append(f"• 可用的splits: {self.available_splits}")
            elif "Config name is missing" in str(e):
                # 从错误信息中提取可用的configs
                suggestions.append(f"• 数据集需要指定config名称")
                suggestions.append(f"• 当前config设置: {self.config or 'None'}")
                suggestions.append(f"• 请在config文件中为此benchmark指定正确的config名称")
                # 尝试提取可用configs列表
                import re
                configs_match = re.search(r"available configs: (\[.*?\])", str(e))
                if configs_match:
                    suggestions.append(f"• 可用的configs: {configs_match.group(1)}")
            elif "split" in str(e).lower():
                suggestions.append(f"• 当前使用的split: {self.split}")
                if self.available_splits:
                    suggestions.append(f"• 可用的splits: {self.available_splits}")
                suggestions.append(f"• 尝试使用其他split，或在config中更新available_splits")
            
            if suggestions:
                error_msg += "\n建议:\n" + "\n".join(suggestions)
            
            raise RuntimeError(error_msg)
    
    def _convert_to_task(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkTask]:
        """
        将数据集item转换为BenchmarkTask
        
        不同benchmark的数据格式可能不同，需要适配
        """
        task_id = item.get("id", item.get("question_id", item.get("questionId", f"task_{idx}")))
        
        # 获取问题 - 支持多种字段名
        question = item.get("question", item.get("text", item.get("query", "")))
        if not question:
            return None
        
        # 获取图像 - 支持多种字段名
        images = []
        image_fields = ["image", "img", "image_path", "image_file", "imageId"]
        for field in image_fields:
            if field in item and item[field] is not None:
                images.append(item[field])
                break  # 只取第一个找到的图像字段
        
        # 获取正确答案 - 支持多种字段名
        ground_truth = None
        answer_fields = ["answer", "answers", "label", "target", "gt_answer"]
        for field in answer_fields:
            if field in item and item[field] is not None:
                ground_truth = item[field]
                break
        
        # 如果ground_truth是列表，取第一个
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            ground_truth = ground_truth[0]
        
        # 提取metadata（包括taxonomy信息）
        metadata = {}
        
        # 1. 首先提取config中定义的native_taxonomy_fields
        taxonomy_fields = self.benchmark_info.get("native_taxonomy_fields", [])
        for field in taxonomy_fields:
            if field in item:
                metadata[field] = item[field]
        
        # 2. 然后提取其他常见的有用字段（如果不在taxonomy中）
        additional_fields = [
            "question_type", "semantic", "program", "question_family", 
            "chart_type", "category", "capability", "task_group",
            "difficulty", "skill", "domain", "task", "subcategory"
        ]
        for key in additional_fields:
            if key in item and key not in metadata:
                metadata[key] = item[key]
        
        # 3. 添加benchmark配置中的note（如果有）
        if "note" in self.benchmark_info:
            metadata["benchmark_note"] = self.benchmark_info["note"]
        
        return BenchmarkTask(
            task_id=str(task_id),
            question=str(question),
            images=images if images else [],
            ground_truth=ground_truth,
            metadata=metadata
        )
    
    def evaluate_answer(self, 
                       model_answer: str, 
                       ground_truth: Any,
                       task: BenchmarkTask) -> BenchmarkResult:
        """
        评估模型答案
        
        简单实现：字符串匹配
        可以根据不同benchmark的特点实现更复杂的评估逻辑
        """
        model_answer_clean = str(model_answer).strip().lower()
        ground_truth_clean = str(ground_truth).strip().lower()
        
        # 精确匹配
        is_correct = model_answer_clean == ground_truth_clean
        
        # 部分匹配（如果答案包含关键词）
        if not is_correct:
            if ground_truth_clean in model_answer_clean or model_answer_clean in ground_truth_clean:
                score = 0.5
            else:
                score = 0.0
        else:
            score = 1.0
        
        # 可以在这里实现特定benchmark的评估逻辑
        # 例如GQA可能需要特殊处理，CLEVR需要执行程序等
        
        return BenchmarkResult(
            task_id=task.task_id,
            question=task.question,
            ground_truth=ground_truth,
            model_answer=model_answer,
            is_correct=is_correct,
            score=score,
            metadata={"evaluation_method": "exact_match", "benchmark": self.name}
        )
    
    def get_info(self) -> Dict[str, Any]:
        """获取benchmark信息"""
        info = super().get_info()
        info.update({
            "hf_id": self.hf_id,
            "config": self.config,
            "split": self.split,
            "available_splits": self.available_splits,
            "source": "huggingface",
            "native_taxonomy_fields": self.benchmark_info.get("native_taxonomy_fields", []),
            "note": self.benchmark_info.get("note", "")
        })
        return info
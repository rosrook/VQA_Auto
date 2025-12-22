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
#             hf_id: HuggingFace数据集ID（完整ID，如 "lmms-lab/GQA"）
#             config: 数据集配置名（如 "balanced", "default"）
#             split: 数据集split（如 "validation", "test"），如果为None则使用默认split
#             **kwargs: 其他参数，包括 benchmark_info
#         """
#         self.hf_id = hf_id
#         self.config = config if config and config != "default" else None
#         self.benchmark_info = kwargs.get("benchmark_info", {})
        
#         # 如果没有指定split，使用benchmark_info中的默认split
#         if split is None:
#             split = self.benchmark_info.get("default_split", "validation")
#         self.split = split
        
#         # 获取可用的splits（用于验证）
#         self.available_splits = self.benchmark_info.get("available_splits", [])
        
#         # data_path参数在这里不使用，但需要传递以满足基类要求
#         data_path = f"hf://{self.hf_id}"
#         if self.config:
#             data_path += f"/{self.config}"
#         data_path += f"/{self.split}"
        
#         super().__init__(name, data_path, **kwargs)
#         self.description = f"HuggingFace Benchmark: {name} ({self.hf_id})"
    
#     # def _load_data(self):
#     #     """从HuggingFace Hub加载数据"""
#     #     try:
#     #         from datasets import load_dataset
#     #     except ImportError:
#     #         raise ImportError(
#     #             "需要安装datasets库: pip install datasets"
#     #         )
        
#     #     try:
#     #         print(f"🔄 正在加载数据集: {self.hf_id}")
#     #         print(f"   配置: {self.config or 'default'}, Split: {self.split}")
            
#     #         # 验证split是否可用
#     #         if self.available_splits and self.split not in self.available_splits:
#     #             print(f"⚠️  警告: split '{self.split}' 不在可用列表中 {self.available_splits}")
            
#     #         # 加载数据集
#     #         load_kwargs = {
#     #             "split": self.split,
#     #             "trust_remote_code": True
#     #         }
            
#     #         if self.config:
#     #             load_kwargs["name"] = self.config
            
#     #         dataset = load_dataset(self.hf_id, **load_kwargs)
            
#     #         print(f"✓ 成功加载 {len(dataset)} 条数据")
            
#     #         self.tasks = []
            
#     #         # 将数据集转换为BenchmarkTask列表
#     #         for idx, item in enumerate(dataset):
#     #             task = self._convert_to_task(item, idx)
#     #             if task:
#     #                 self.tasks.append(task)
            
#     #         print(f"✓ 转换完成，共 {len(self.tasks)} 个任务")
        
#     #     except Exception as e:
#     #         # 提供更详细的错误信息
#     #         error_msg = f"从HuggingFace加载数据失败 ({self.hf_id}): {e}"
#     #         print(f"✗ {error_msg}")
            
#     #         # 提供可能的解决方案
#     #         suggestions = []
#     #         if "doesn't exist" in str(e) or "cannot be accessed" in str(e):
#     #             suggestions.append(f"• 检查数据集ID是否正确: {self.hf_id}")
#     #             suggestions.append(f"• 访问 https://huggingface.co/datasets/{self.hf_id} 确认数据集存在")
#     #             if self.config:
#     #                 suggestions.append(f"• 检查配置名称是否正确: {self.config}")
#     #             suggestions.append(f"• 检查split是否正确: {self.split}")
#     #             if self.available_splits:
#     #                 suggestions.append(f"• 可用的splits: {self.available_splits}")
#     #         elif "split" in str(e).lower():
#     #             suggestions.append(f"• 当前使用的split: {self.split}")
#     #             if self.available_splits:
#     #                 suggestions.append(f"• 可用的splits: {self.available_splits}")
#     #             suggestions.append(f"• 尝试使用其他split，或在config中更新available_splits")
            
#     #         if suggestions:
#     #             error_msg += "\n建议:\n" + "\n".join(suggestions)
            
#     #         raise RuntimeError(error_msg)

#     def _load_data(self):
#         """从HuggingFace Hub加载数据"""
#         try:
#             from datasets import load_dataset
#         except ImportError:
#             raise ImportError(
#                 "需要安装datasets库: pip install datasets"
#             )
        
#         try:
#             print(f"🔄 正在加载数据集: {self.hf_id}")
#             print(f"   配置: {self.config or 'default'}, Split: {self.split}")
            
#             # 验证split是否可用
#             if self.available_splits and self.split not in self.available_splits:
#                 print(f"⚠️  警告: split '{self.split}' 不在可用列表中 {self.available_splits}")
            
#             # 加载数据集
#             load_kwargs = {
#                 "split": self.split
#             }
            
#             if self.config:
#                 load_kwargs["name"] = self.config
            
#             dataset = load_dataset(self.hf_id, **load_kwargs)
            
#             print(f"✓ 成功加载 {len(dataset)} 条数据")
            
#             self.tasks = []
            
#             # 将数据集转换为BenchmarkTask列表
#             for idx, item in enumerate(dataset):
#                 task = self._convert_to_task(item, idx)
#                 if task:
#                     self.tasks.append(task)
            
#             print(f"✓ 转换完成，共 {len(self.tasks)} 个任务")
        
#         except Exception as e:
#             # 提供更详细的错误信息
#             error_msg = f"从HuggingFace加载数据失败 ({self.hf_id}): {e}"
#             print(f"✗ {error_msg}")
            
#             # 提供可能的解决方案
#             suggestions = []
#             if "doesn't exist" in str(e) or "cannot be accessed" in str(e):
#                 suggestions.append(f"• 检查数据集ID是否正确: {self.hf_id}")
#                 suggestions.append(f"• 访问 https://huggingface.co/datasets/{self.hf_id} 确认数据集存在")
#                 if self.config:
#                     suggestions.append(f"• 检查配置名称是否正确: {self.config}")
#                 suggestions.append(f"• 检查split是否正确: {self.split}")
#                 if self.available_splits:
#                     suggestions.append(f"• 可用的splits: {self.available_splits}")
#             elif "Config name is missing" in str(e):
#                 # 从错误信息中提取可用的configs
#                 suggestions.append(f"• 数据集需要指定config名称")
#                 suggestions.append(f"• 当前config设置: {self.config or 'None'}")
#                 suggestions.append(f"• 请在config文件中为此benchmark指定正确的config名称")
#                 # 尝试提取可用configs列表
#                 import re
#                 configs_match = re.search(r"available configs: (\[.*?\])", str(e))
#                 if configs_match:
#                     suggestions.append(f"• 可用的configs: {configs_match.group(1)}")
#             elif "split" in str(e).lower():
#                 suggestions.append(f"• 当前使用的split: {self.split}")
#                 if self.available_splits:
#                     suggestions.append(f"• 可用的splits: {self.available_splits}")
#                 suggestions.append(f"• 尝试使用其他split，或在config中更新available_splits")
            
#             if suggestions:
#                 error_msg += "\n建议:\n" + "\n".join(suggestions)
            
#             raise RuntimeError(error_msg)
    
#     def _convert_to_task(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkTask]:
#         """
#         将数据集item转换为BenchmarkTask
        
#         不同benchmark的数据格式可能不同，需要适配
#         """
#         task_id = item.get("id", item.get("question_id", item.get("questionId", f"task_{idx}")))
        
#         # 获取问题 - 支持多种字段名
#         question = item.get("question", item.get("text", item.get("query", "")))
#         if not question:
#             return None
        
#         # 获取图像 - 支持多种字段名
#         images = []
#         image_fields = ["image", "img", "image_path", "image_file", "imageId"]
#         for field in image_fields:
#             if field in item and item[field] is not None:
#                 images.append(item[field])
#                 break  # 只取第一个找到的图像字段
        
#         # 获取正确答案 - 支持多种字段名
#         ground_truth = None
#         answer_fields = ["answer", "answers", "label", "target", "gt_answer"]
#         for field in answer_fields:
#             if field in item and item[field] is not None:
#                 ground_truth = item[field]
#                 break
        
#         # 如果ground_truth是列表，取第一个
#         if isinstance(ground_truth, list) and len(ground_truth) > 0:
#             ground_truth = ground_truth[0]
        
#         # 提取metadata（包括taxonomy信息）
#         metadata = {}
        
#         # 1. 首先提取config中定义的native_taxonomy_fields
#         taxonomy_fields = self.benchmark_info.get("native_taxonomy_fields", [])
#         for field in taxonomy_fields:
#             if field in item:
#                 metadata[field] = item[field]
        
#         # 2. 然后提取其他常见的有用字段（如果不在taxonomy中）
#         additional_fields = [
#             "question_type", "semantic", "program", "question_family", 
#             "chart_type", "category", "capability", "task_group",
#             "difficulty", "skill", "domain", "task", "subcategory"
#         ]
#         for key in additional_fields:
#             if key in item and key not in metadata:
#                 metadata[key] = item[key]
        
#         # 3. 添加benchmark配置中的note（如果有）
#         if "note" in self.benchmark_info:
#             metadata["benchmark_note"] = self.benchmark_info["note"]
        
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
#             "available_splits": self.available_splits,
#             "source": "huggingface",
#             "native_taxonomy_fields": self.benchmark_info.get("native_taxonomy_fields", []),
#             "note": self.benchmark_info.get("note", "")
#         })
#         return info






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
                 load_data_on_init: bool = False,
                 **kwargs):
        """
        Args:
            name: Benchmark名称
            hf_id: HuggingFace数据集ID（完整ID，如 "lmms-lab/GQA"）
            config: 数据集配置名（如 "balanced", "default"）
            split: 数据集split（如 "validation", "test"），如果为None则使用默认split
            load_data_on_init: 是否在初始化时加载数据（False表示延迟加载，使用流式）
            **kwargs: 其他参数，包括 benchmark_info
        """
        self.hf_id = hf_id
        self.config = config if config and config != "default" else None
        self.benchmark_info = kwargs.get("benchmark_info", {})
        self._dataset = None  # 延迟加载数据集
        self._use_streaming = not load_data_on_init  # 使用流式加载
        
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
        
        super().__init__(name, data_path, load_data_on_init=load_data_on_init, **kwargs)
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

    def _get_dataset(self):
        """获取数据集（延迟加载，支持流式）"""
        if self._dataset is not None:
            return self._dataset
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "需要安装datasets库: pip install datasets"
            )
        
        try:
            print(f"🔄 正在加载数据集: {self.hf_id}")
            print(f"   配置: {self.config or 'default'}, Split: {self.split}")
            print(f"   模式: {'流式加载' if self._use_streaming else '批量加载'}")
            
            # 验证split是否可用
            if self.available_splits and self.split not in self.available_splits:
                print(f"⚠️  警告: split '{self.split}' 不在可用列表中 {self.available_splits}")
            
            # 特殊处理：对于 GQA 的 testdev_balanced_images 配置，需要同时加载问题和答案
            # testdev_balanced_images 只包含图像，需要从其他配置获取问题和答案
            if self.hf_id == "lmms-lab/GQA" and self.config == "testdev_balanced_images":
                print(f"  ℹ️  检测到 GQA testdev_balanced_images 配置，将同时加载问题和答案数据")
                # 加载图像数据
                image_kwargs = {
                    "split": self.split,
                    "streaming": self._use_streaming,
                    "name": self.config
                }
                print(f"   加载图像数据，参数: {image_kwargs}")
                image_dataset = load_dataset(self.hf_id, **image_kwargs)
                
                # 尝试加载包含问题和答案的数据
                # 根据配置名称推断对应的 instructions 配置
                # testdev_balanced_images -> testdev_balanced_instructions
                base_config = self.config.replace("_images", "")
                print(f"  🔍 基础配置: {base_config} (从 {self.config} 提取)")
                
                # 构建可能的配置列表
                question_configs = [
                    f"{base_config}_instructions",  # testdev_balanced_instructions
                ]
                # 如果base_config包含_balanced，也尝试_all_instructions
                if "_balanced" in base_config:
                    split_name = base_config.split("_")[0]  # testdev
                    question_configs.append(f"{split_name}_all_instructions")  # testdev_all_instructions
                
                print(f"  🔍 将尝试以下配置: {question_configs}")
                question_dataset = None
                
                for q_config in question_configs:
                    try:
                        # 对于 instructions 配置，可能需要不同的 split 处理
                        # 先尝试使用相同的 split，如果失败，尝试不使用 split（让数据集自己决定）
                        question_kwargs = {
                            "split": self.split,
                            "streaming": self._use_streaming,
                            "name": q_config
                        }
                        
                        print(f"   尝试加载问题和答案数据，配置: {q_config}, 参数: {question_kwargs}")
                        try:
                            temp_dataset = load_dataset(self.hf_id, **question_kwargs)
                        except Exception as split_error:
                            # 如果使用 split 失败，尝试不使用 split
                            print(f"   使用 split={self.split} 失败，尝试不使用 split: {split_error}")
                            question_kwargs_no_split = {
                                "streaming": self._use_streaming,
                                "name": q_config
                            }
                            temp_dataset = load_dataset(self.hf_id, **question_kwargs_no_split)
                            # 如果成功，更新 question_kwargs 以便后续使用
                            question_kwargs = question_kwargs_no_split
                        
                        # 检查第一个item是否包含question字段
                        if self._use_streaming:
                            # 流式模式：创建迭代器并检查第一个item
                            temp_iter = iter(temp_dataset)
                            test_item = next(temp_iter)
                            # 重新创建数据集（因为迭代器已被消耗）
                            question_dataset = load_dataset(self.hf_id, **question_kwargs)
                        else:
                            # 非流式模式：直接检查
                            test_item = temp_dataset[0] if len(temp_dataset) > 0 else {}
                            question_dataset = temp_dataset
                        
                        # 检查是否包含问题相关的字段
                        has_question_field = any(field in test_item for field in [
                            "question", "text", "sent", "sentence", "instruction", 
                            "prompt", "input", "query", "question_text"
                        ])
                        # 检查是否包含答案字段
                        has_answer_field = any(field in test_item for field in [
                            "answer", "answers", "label", "target", "gt_answer", "ground_truth"
                        ])
                        
                        if has_question_field or has_answer_field:
                            print(f"  ✓ 找到包含问题和答案的配置: {q_config}")
                            print(f"     包含的字段: {list(test_item.keys())}")
                            break
                        else:
                            print(f"  ⚠️  配置 {q_config} 不包含问题或答案字段，字段: {list(test_item.keys())}")
                            question_dataset = None
                    except Exception as e:
                        print(f"  ⚠️  配置 {q_config or 'default'} 加载失败: {e}")
                        question_dataset = None
                        continue
                
                if question_dataset:
                    # 合并数据集：将问题和答案合并到图像数据中
                    self._dataset = self._merge_gqa_datasets(image_dataset, question_dataset)
                    print(f"✓ GQA 数据集合并完成")
                else:
                    print(f"  ⚠️  无法找到包含问题和答案的配置")
                    print(f"  ℹ️  将仅使用图像数据（可能无法获取问题和答案）")
                    self._dataset = image_dataset
            else:
                # 普通加载方式
                load_kwargs = {
                    "split": self.split,
                    "streaming": self._use_streaming
                }
                
                if self.config:
                    load_kwargs["name"] = self.config
                
                print(f"   加载参数: {load_kwargs}")
                self._dataset = load_dataset(self.hf_id, **load_kwargs)
            
            if not self._use_streaming:
                print(f"✓ 成功加载 {len(self._dataset)} 条数据")
            else:
                print(f"✓ 流式数据集已就绪")
            
            return self._dataset
        
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
                suggestions.append(f"• 数据集需要指定config名称")
                suggestions.append(f"• 当前config设置: {self.config or 'None'}")
                suggestions.append(f"• 请在config文件中为此benchmark指定正确的config名称")
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
    
    def _merge_gqa_datasets(self, image_dataset, question_dataset):
        """
        合并 GQA 的图像数据集和问题数据集
        
        Args:
            image_dataset: 包含图像的数据集（只有 id 和 image）
            question_dataset: 包含问题和答案的数据集（有 id, question, answer 等）
        
        Returns:
            合并后的数据集迭代器
        """
        if self._use_streaming:
            # 流式模式：创建一个合并迭代器
            # 对于流式数据，我们需要先构建一个 ID 到问题的映射
            # 但由于是流式的，我们需要边迭代边匹配
            
            class MergedDatasetIterator:
                def __init__(self, image_iter, question_iter):
                    self.image_iter = iter(image_iter)
                    self.question_iter = iter(question_iter)
                    # 预加载一些问题数据到内存（用于匹配）
                    self.question_cache = {}
                    self._preload_questions()
                
                def _preload_questions(self):
                    """预加载一些问题数据到缓存"""
                    try:
                        count = 0
                        for item in self.question_iter:
                            # 使用 imageId 作为键，因为图像数据集的 id 对应问题数据集的 imageId
                            image_id = item.get("imageId", item.get("image_id", ""))
                            if image_id:
                                # 如果同一个图像有多个问题，存储为列表
                                if image_id not in self.question_cache:
                                    self.question_cache[image_id] = []
                                self.question_cache[image_id].append(item)
                                count += 1
                            # 限制缓存大小，避免内存溢出
                            if count >= 50000:  # 增加缓存大小以覆盖更多数据
                                break
                        print(f"  ✓ 预加载了 {count} 个问题到缓存，覆盖 {len(self.question_cache)} 个图像")
                    except StopIteration:
                        print(f"  ✓ 预加载完成，共 {count} 个问题，覆盖 {len(self.question_cache)} 个图像")
                    except Exception as e:
                        print(f"  ⚠️  预加载问题时出错: {e}")
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    # 获取下一个图像项
                    image_item = next(self.image_iter)
                    image_id = image_item.get("id", "")
                    
                    # 尝试从缓存中找到匹配的问题（使用 imageId 匹配）
                    if image_id in self.question_cache:
                        question_items = self.question_cache[image_id]
                        # 如果有多个问题，取第一个（或者可以随机选择）
                        question_item = question_items[0] if isinstance(question_items, list) else question_items
                        # 合并数据
                        merged_item = {**image_item}
                        merged_item.update(question_item)
                        return merged_item
                    else:
                        # 如果缓存中没有匹配的问题，返回图像项（但会因为没有question而被跳过）
                        return image_item
            
            return MergedDatasetIterator(image_dataset, question_dataset)
        else:
            # 非流式模式：构建 ID 映射并合并
            # 使用 imageId 作为键，因为图像数据集的 id 对应问题数据集的 imageId
            question_dict = {}
            for item in question_dataset:
                image_id = item.get("imageId", item.get("image_id", ""))
                if image_id:
                    # 如果同一个图像有多个问题，存储为列表
                    if image_id not in question_dict:
                        question_dict[image_id] = []
                    question_dict[image_id].append(item)
            
            # 合并数据
            merged_data = []
            for image_item in image_dataset:
                image_id = image_item.get("id", "")
                merged_item = {**image_item}
                if image_id in question_dict:
                    # 如果有多个问题，取第一个
                    question_items = question_dict[image_id]
                    question_item = question_items[0] if isinstance(question_items, list) else question_items
                    merged_item.update(question_item)
                merged_data.append(merged_item)
            
            return merged_data
    
    def _load_data(self):
        """从HuggingFace Hub加载数据（一次性加载所有，用于兼容性）"""
        dataset = self._get_dataset()
        
        if self._use_streaming:
            # 流式加载模式，不预加载所有数据
            return
        
        self.tasks = []
        
        # 将数据集转换为BenchmarkTask列表
        for idx, item in enumerate(dataset):
            task = self._convert_to_task(item, idx)
            if task:
                self.tasks.append(task)
        
        print(f"✓ 转换完成，共 {len(self.tasks)} 个任务")
    
    def get_dataset_iterator(self):
        """
        获取数据集迭代器（用于流式处理）
        
        Returns:
            数据集迭代器
        """
        dataset = self._get_dataset()
        
        if self._use_streaming:
            # 流式数据集，直接返回迭代器
            if hasattr(dataset, '__iter__'):
                return iter(dataset)
            else:
                # 如果不是迭代器，尝试转换为迭代器
                return iter(dataset)
        else:
            # 普通数据集，返回列表迭代器
            if self.tasks:
                return iter(self.tasks)
            else:
                # 如果没有预加载的任务，从数据集创建
                return iter(dataset)
    
    def get_task_from_item(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkTask]:
        """
        从数据集item创建BenchmarkTask（用于流式处理）
        
        Args:
            item: 数据集中的一个item
            idx: 索引
        
        Returns:
            BenchmarkTask或None
        """
        return self._convert_to_task(item, idx)
    
    def _convert_to_task(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkTask]:
        """
        将数据集item转换为BenchmarkTask
        
        不同benchmark的数据格式可能不同，需要适配
        """
        task_id = item.get("id", item.get("question_id", item.get("questionId", f"task_{idx}")))
        
        # 获取问题 - 支持多种字段名（包括GQA可能使用的字段）
        question = None
        question_fields = ["question", "text", "query", "sent", "sentence", "prompt", "input"]
        for field in question_fields:
            if field in item and item[field] is not None:
                question_value = item[field]
                # 如果是字符串且非空，使用它
                if isinstance(question_value, str) and question_value.strip():
                    question = question_value
                    break
                # 如果是列表，取第一个非空字符串
                elif isinstance(question_value, list) and len(question_value) > 0:
                    first_item = question_value[0]
                    if isinstance(first_item, str) and first_item.strip():
                        question = first_item
                        break
        
        if not question:
            # 调试：打印item的键以帮助诊断问题
            if idx < 5:  # 只打印前5个，避免输出过多
                print(f"  🔍 调试: item (idx={idx}) 的字段: {list(item.keys())}")
                # 打印所有可能包含文本的字段的值
                for key in item.keys():
                    value = item[key]
                    if isinstance(value, str) and len(value) > 0:
                        print(f"    - {key}: {value[:100]}...")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"    - {key}: {type(value[0])} list with {len(value)} items")
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
            "note": self.benchmark_info.get("note", ""),
            "use_streaming": self._use_streaming
        })
        
        # 如果是流式加载，num_tasks可能未知
        if self._use_streaming and self._dataset is None:
            info["num_tasks"] = "unknown (streaming)"
        
        return info
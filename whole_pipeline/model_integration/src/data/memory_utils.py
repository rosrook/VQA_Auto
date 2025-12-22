"""
内存管理和优化工具
提供数据集大小估算、内存监控等功能
"""
import psutil
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json
import gc
import sys

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """
        获取当前内存使用情况
        
        Returns:
            包含内存信息的字典 (单位: GB)
        """
        process = psutil.Process()
        
        # 系统内存
        system_memory = psutil.virtual_memory()
        
        # 进程内存
        process_memory = process.memory_info()
        
        # GPU内存（如果可用）
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i) / 1024**3
                gpu_memory[f'gpu_{i}_reserved'] = torch.cuda.memory_reserved(i) / 1024**3
                gpu_memory[f'gpu_{i}_total'] = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        return {
            'system_total_gb': system_memory.total / 1024**3,
            'system_available_gb': system_memory.available / 1024**3,
            'system_used_gb': system_memory.used / 1024**3,
            'system_percent': system_memory.percent,
            'process_rss_gb': process_memory.rss / 1024**3,
            'process_vms_gb': process_memory.vms / 1024**3,
            **gpu_memory
        }
    
    @staticmethod
    def print_memory_info():
        """打印内存信息"""
        info = MemoryMonitor.get_memory_info()
        
        logger.info("=" * 60)
        logger.info("内存使用情况")
        logger.info("=" * 60)
        logger.info(f"系统总内存: {info['system_total_gb']:.2f} GB")
        logger.info(f"系统可用内存: {info['system_available_gb']:.2f} GB")
        logger.info(f"系统已用内存: {info['system_used_gb']:.2f} GB ({info['system_percent']:.1f}%)")
        logger.info(f"进程RSS内存: {info['process_rss_gb']:.2f} GB")
        
        if torch.cuda.is_available():
            logger.info("-" * 60)
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}:")
                logger.info(f"  已分配: {info[f'gpu_{i}_allocated']:.2f} GB")
                logger.info(f"  已保留: {info[f'gpu_{i}_reserved']:.2f} GB")
                logger.info(f"  总容量: {info[f'gpu_{i}_total']:.2f} GB")
        
        logger.info("=" * 60)
    
    @staticmethod
    def check_memory_available(required_gb: float) -> bool:
        """
        检查是否有足够的可用内存
        
        Args:
            required_gb: 需要的内存大小 (GB)
            
        Returns:
            是否有足够内存
        """
        info = MemoryMonitor.get_memory_info()
        available = info['system_available_gb']
        
        if available < required_gb:
            logger.warning(
                f"内存不足！需要 {required_gb:.2f} GB，"
                f"但只有 {available:.2f} GB 可用"
            )
            return False
        
        return True
    
    @staticmethod
    def clear_memory():
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("内存清理完成")


class DatasetSizeEstimator:
    """数据集大小估算器"""
    
    @staticmethod
    def estimate_image_memory(
        num_images: int,
        image_size: tuple = (224, 224),
        channels: int = 3,
        dtype: str = 'float32'
    ) -> float:
        """
        估算图像占用的内存
        
        Args:
            num_images: 图像数量
            image_size: 图像尺寸 (H, W)
            channels: 通道数
            dtype: 数据类型
            
        Returns:
            估算的内存大小 (GB)
        """
        bytes_per_element = {
            'float32': 4,
            'float16': 2,
            'uint8': 1
        }
        
        bytes_per_image = (
            image_size[0] * image_size[1] * channels * 
            bytes_per_element.get(dtype, 4)
        )
        
        total_bytes = num_images * bytes_per_image
        total_gb = total_bytes / 1024**3
        
        return total_gb
    
    @staticmethod
    def estimate_text_memory(
        num_samples: int,
        max_length: int = 512,
        dtype: str = 'int64'
    ) -> float:
        """
        估算文本数据占用的内存
        
        Args:
            num_samples: 样本数量
            max_length: 最大长度
            dtype: 数据类型
            
        Returns:
            估算的内存大小 (GB)
        """
        bytes_per_element = {
            'int64': 8,
            'int32': 4,
            'int16': 2
        }
        
        # input_ids + attention_mask
        bytes_per_sample = 2 * max_length * bytes_per_element.get(dtype, 8)
        total_bytes = num_samples * bytes_per_sample
        total_gb = total_bytes / 1024**3
        
        return total_gb
    
    @staticmethod
    def estimate_vqa_dataset_memory(
        num_samples: int,
        image_size: tuple = (224, 224),
        max_question_length: int = 128,
        max_answer_length: int = 32,
        preload_images: bool = False
    ) -> Dict[str, float]:
        """
        估算VQA数据集的内存占用
        
        Args:
            num_samples: 样本数量
            image_size: 图像尺寸
            max_question_length: 最大问题长度
            max_answer_length: 最大答案长度
            preload_images: 是否预加载图像
            
        Returns:
            详细的内存估算
        """
        # 图像内存
        if preload_images:
            image_memory = DatasetSizeEstimator.estimate_image_memory(
                num_samples, image_size
            )
        else:
            image_memory = 0  # 懒加载不占用内存
        
        # 问题内存
        question_memory = DatasetSizeEstimator.estimate_text_memory(
            num_samples, max_question_length
        )
        
        # 答案内存
        answer_memory = DatasetSizeEstimator.estimate_text_memory(
            num_samples, max_answer_length
        )
        
        # 元数据内存（粗略估计）
        metadata_memory = num_samples * 0.001  # 约1KB per sample
        
        total_memory = image_memory + question_memory + answer_memory + metadata_memory
        
        return {
            'total_gb': total_memory,
            'image_gb': image_memory,
            'question_gb': question_memory,
            'answer_gb': answer_memory,
            'metadata_gb': metadata_memory,
            'num_samples': num_samples
        }
    
    @staticmethod
    def print_dataset_estimate(estimate: Dict[str, float]):
        """打印数据集内存估算"""
        logger.info("=" * 60)
        logger.info("数据集内存估算")
        logger.info("=" * 60)
        logger.info(f"样本数量: {estimate['num_samples']:,}")
        logger.info(f"图像内存: {estimate['image_gb']:.2f} GB")
        logger.info(f"问题内存: {estimate['question_gb']:.2f} GB")
        logger.info(f"答案内存: {estimate['answer_gb']:.2f} GB")
        logger.info(f"元数据内存: {estimate['metadata_gb']:.2f} GB")
        logger.info(f"总计: {estimate['total_gb']:.2f} GB")
        logger.info("=" * 60)
    
    @staticmethod
    def recommend_strategy(
        num_samples: int,
        available_memory_gb: float,
        image_size: tuple = (224, 224)
    ) -> str:
        """
        推荐数据加载策略
        
        Args:
            num_samples: 样本数量
            available_memory_gb: 可用内存 (GB)
            image_size: 图像尺寸
            
        Returns:
            推荐策略
        """
        # 估算预加载所需内存
        estimate = DatasetSizeEstimator.estimate_vqa_dataset_memory(
            num_samples, image_size, preload_images=True
        )
        
        required_memory = estimate['total_gb'] * 1.5  # 留50%余量
        
        if required_memory < available_memory_gb * 0.5:
            return "small_dataset"  # 可以预加载
        elif required_memory < available_memory_gb * 2:
            return "medium_dataset"  # 使用懒加载+缓存
        else:
            return "large_dataset"  # 必须使用流式或内存映射


class DatasetOptimizer:
    """数据集优化建议"""
    
    @staticmethod
    def analyze_and_recommend(
        data_path: str,
        image_root: Optional[str] = None,
        batch_size: int = 16
    ) -> Dict:
        """
        分析数据集并给出优化建议
        
        Args:
            data_path: 数据文件路径
            image_root: 图像根目录
            batch_size: 批次大小
            
        Returns:
            优化建议
        """
        from data_loader import DataLoader
        
        logger.info("分析数据集...")
        
        # 加载数据统计
        loader = DataLoader(data_path)
        data = loader.load()
        num_samples = len(data)
        
        # 获取内存信息
        memory_info = MemoryMonitor.get_memory_info()
        available_memory = memory_info['system_available_gb']
        
        # 估算内存需求
        estimate = DatasetSizeEstimator.estimate_vqa_dataset_memory(
            num_samples, preload_images=True
        )
        
        # 推荐策略
        strategy = DatasetSizeEstimator.recommend_strategy(
            num_samples, available_memory
        )
        
        # 生成建议
        recommendations = {
            'num_samples': num_samples,
            'available_memory_gb': available_memory,
            'estimated_memory_gb': estimate['total_gb'],
            'recommended_strategy': strategy,
            'recommendations': []
        }
        
        if strategy == "small_dataset":
            recommendations['recommendations'] = [
                "数据集较小，可以使用标准Dataset",
                f"建议batch_size: {min(batch_size, 32)}",
                "可以考虑预加载图像以加速训练",
                "num_workers可以设置为4-8"
            ]
        
        elif strategy == "medium_dataset":
            recommendations['recommendations'] = [
                "数据集中等大小，建议使用LazyLoadVQADataset",
                f"建议batch_size: {min(batch_size, 16)}",
                "启用图像缓存（cache_images=True）",
                "num_workers设置为4",
                "考虑使用混合精度训练（fp16）"
            ]
        
        else:  # large_dataset
            recommendations['recommendations'] = [
                "数据集很大，必须使用优化策略！",
                "强烈建议使用StreamingVQADataset或MemoryMappedVQADataset",
                f"建议batch_size: {min(batch_size, 8)}",
                "不要预加载图像",
                "num_workers设置为2-4",
                "使用梯度累积来模拟更大的batch",
                "考虑使用混合精度训练（fp16）",
                "如果可能，预处理数据为memmap格式"
            ]
        
        return recommendations
    
    @staticmethod
    def print_recommendations(recommendations: Dict):
        """打印优化建议"""
        logger.info("=" * 60)
        logger.info("数据集优化建议")
        logger.info("=" * 60)
        logger.info(f"样本数量: {recommendations['num_samples']:,}")
        logger.info(f"可用内存: {recommendations['available_memory_gb']:.2f} GB")
        logger.info(f"估算内存: {recommendations['estimated_memory_gb']:.2f} GB")
        logger.info(f"推荐策略: {recommendations['recommended_strategy']}")
        logger.info("-" * 60)
        logger.info("建议:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            logger.info(f"{i}. {rec}")
        logger.info("=" * 60)


class BatchSizeCalculator:
    """批次大小计算器"""
    
    @staticmethod
    def calculate_max_batch_size(
        model_params: int,
        image_size: tuple = (224, 224),
        sequence_length: int = 128,
        available_gpu_memory_gb: float = 16.0,
        safety_factor: float = 0.7
    ) -> int:
        """
        计算最大可用的batch size
        
        Args:
            model_params: 模型参数量（百万）
            image_size: 图像尺寸
            sequence_length: 序列长度
            available_gpu_memory_gb: 可用GPU内存 (GB)
            safety_factor: 安全系数（0-1）
            
        Returns:
            推荐的最大batch size
        """
        # 模型内存 (粗略估计)
        model_memory_gb = model_params * 4 / 1000  # 4 bytes per param
        
        # 激活值内存 (粗略估计)
        activation_per_sample_mb = (
            image_size[0] * image_size[1] * 3 * 4 +  # 图像
            sequence_length * 768 * 4  # 文本特征（假设hidden_size=768）
        ) / 1024**2
        
        # 可用于batch的内存
        available_for_batch_gb = (available_gpu_memory_gb - model_memory_gb) * safety_factor
        available_for_batch_mb = available_for_batch_gb * 1024
        
        # 计算batch size
        max_batch_size = int(available_for_batch_mb / activation_per_sample_mb)
        
        # 向下取整到2的幂
        batch_size = 1
        while batch_size * 2 <= max_batch_size:
            batch_size *= 2
        
        return max(1, batch_size)


# 示例使用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. 内存监控
    MemoryMonitor.print_memory_info()
    
    # 2. 数据集内存估算
    estimate = DatasetSizeEstimator.estimate_vqa_dataset_memory(
        num_samples=100000,
        preload_images=False
    )
    DatasetSizeEstimator.print_dataset_estimate(estimate)
    
    # 3. 推荐策略
    strategy = DatasetSizeEstimator.recommend_strategy(
        num_samples=100000,
        available_memory_gb=32
    )
    logger.info(f"推荐策略: {strategy}")
    
    # 4. Batch size计算
    max_bs = BatchSizeCalculator.calculate_max_batch_size(
        model_params=110,  # 110M参数
        available_gpu_memory_gb=16
    )
    logger.info(f"推荐最大batch size: {max_bs}")
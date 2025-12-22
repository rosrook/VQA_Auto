"""
数据处理器模块
支持多种processor（BLIP、CLIP、ViT等）的数据预处理
与data_pipeline.py集成，提供统一的数据处理接口
"""
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import random
from collections import Counter
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """统一的数据处理器，支持多种processor"""
    
    # 支持的processor类型映射（使用官方推荐的Auto类）
    PROCESSOR_REGISTRY = {
        'blip': {
            'processor_class': AutoProcessor,  # 官方推荐：AutoProcessor会自动选择BlipProcessor
            'image_processor_class': AutoImageProcessor,  # 官方推荐：AutoImageProcessor会自动选择BlipImageProcessor
            'tokenizer_class': AutoTokenizer,  # 官方推荐：AutoTokenizer会自动选择BlipTokenizer
            'model_ids': [
                'Salesforce/blip-vqa-base',
                'Salesforce/blip-vqa-capfilt-large',
                'Salesforce/blip-image-captioning-base',
                'Salesforce/blip-image-captioning-large'
            ]
        },
        'clip': {
            'processor_class': AutoProcessor,  # 官方推荐：AutoProcessor会自动选择CLIPProcessor
            'image_processor_class': AutoImageProcessor,  # 官方推荐：AutoImageProcessor会自动选择CLIPImageProcessor
            'tokenizer_class': AutoTokenizer,  # 官方推荐：AutoTokenizer会自动选择CLIPTokenizer
            'model_ids': [
                'openai/clip-vit-base-patch32',
                'openai/clip-vit-base-patch16',
                'openai/clip-vit-large-patch14'
            ]
        },
        'auto': {
            'processor_class': AutoProcessor,
            'image_processor_class': AutoImageProcessor,
            'tokenizer_class': AutoTokenizer,
            'model_ids': []  # 支持所有模型
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典，包含processor相关配置
                - processor_type: 'blip', 'clip', 'auto' 等
                - processor_name: HuggingFace模型ID或路径
                - batch_preprocess: 是否批量预处理
                - batch_size: 批处理大小
                - num_proc: 并行处理进程数
                - cache_processed: 是否缓存处理结果
                - cache_dir: 缓存目录
        """
        self.config = config or {}
        self.processor_type = self.config.get('processor_type', 'auto')
        self.processor_name = self.config.get('processor_name', None)
        self.processor = None
        self.image_processor = None
        self.tokenizer = None
        
        # 预处理配置
        self.batch_preprocess = self.config.get('batch_preprocess', False)
        self.batch_size = self.config.get('batch_size', 1000)
        self.num_proc = self.config.get('num_proc', 1)
        self.cache_processed = self.config.get('cache_processed', False)
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        
        # 如果指定了processor_name，自动加载
        if self.processor_name:
            self.load_processor(self.processor_name, self.processor_type)
    
    def load_processor(
        self, 
        processor_name: str, 
        processor_type: Optional[str] = None
    ) -> None:
        """
        加载processor
        
        Args:
            processor_name: HuggingFace模型ID或路径
            processor_type: processor类型 ('blip', 'clip', 'auto')
                          如果为None，会自动检测
        """
        if processor_type is None:
            processor_type = self._detect_processor_type(processor_name)
        
        self.processor_type = processor_type
        self.processor_name = processor_name
        
        logger.info(f"加载processor: {processor_name} (类型: {processor_type})")
        
        try:
            processor_info = self.PROCESSOR_REGISTRY.get(processor_type, self.PROCESSOR_REGISTRY['auto'])
            processor_class = processor_info['processor_class']
            
            # 尝试加载完整的processor（包含图像和文本处理）
            try:
                self.processor = processor_class.from_pretrained(processor_name)
                logger.info(f"成功加载 {processor_type} processor")
                
                # 提取image_processor和tokenizer（如果processor包含它们）
                if hasattr(self.processor, 'image_processor'):
                    self.image_processor = self.processor.image_processor
                if hasattr(self.processor, 'tokenizer'):
                    self.tokenizer = self.processor.tokenizer
                    
            except Exception as e:
                logger.warning(f"无法加载完整processor: {e}，尝试分别加载")
                # 分别加载image_processor和tokenizer
                image_processor_class = processor_info['image_processor_class']
                tokenizer_class = processor_info['tokenizer_class']
                
                try:
                    self.image_processor = image_processor_class.from_pretrained(processor_name)
                    logger.info(f"成功加载 {processor_type} image processor")
                except Exception as e2:
                    logger.warning(f"无法加载image processor: {e2}")
                
                try:
                    self.tokenizer = tokenizer_class.from_pretrained(processor_name)
                    logger.info(f"成功加载 {processor_type} tokenizer")
                except Exception as e3:
                    logger.warning(f"无法加载tokenizer: {e3}")
        
        except Exception as e:
            logger.error(f"加载processor失败: {e}")
            raise RuntimeError(f"无法加载processor {processor_name}: {e}")
    
    def _detect_processor_type(self, processor_name: str) -> str:
        """
        根据模型名称自动检测processor类型
        
        Args:
            processor_name: 模型名称
            
        Returns:
            processor类型
        """
        processor_name_lower = processor_name.lower()
        
        # 检查是否匹配已知的模型ID
        for proc_type, proc_info in self.PROCESSOR_REGISTRY.items():
            if proc_type == 'auto':
                continue
            for model_id in proc_info['model_ids']:
                if model_id.lower() in processor_name_lower or processor_name_lower in model_id.lower():
                    return proc_type
        
        # 根据关键词检测
        if 'blip' in processor_name_lower:
            return 'blip'
        elif 'clip' in processor_name_lower:
            return 'clip'
        else:
            return 'auto'  # 默认使用AutoProcessor
    
    def process(
        self, 
        data: List[Dict], 
        task_type: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        处理数据
        
        Args:
            data: 原始数据列表
            task_type: 任务类型 ('vqa', 'image_captioning', 'classification', 等)
                      如果为None，会从config中获取或自动检测
            **kwargs: 其他处理参数
            
        Returns:
            处理后的数据列表
        """
        if not data:
            logger.warning("数据为空，跳过处理")
            return data
        
        # 如果没有指定task_type，尝试从config获取或自动检测
        if task_type is None:
            task_type = self.config.get('task_type')
            if task_type is None:
                task_type = self._detect_task_type(data)
        
        logger.info(f"开始处理 {len(data)} 条数据，任务类型: {task_type}")
        
        # 根据任务类型选择处理方法
        if task_type == 'vqa':
            return self._process_vqa_data(data, **kwargs)
        elif task_type == 'image_captioning':
            return self._process_image_captioning_data(data, **kwargs)
        elif task_type == 'classification':
            return self._process_classification_data(data, **kwargs)
        else:
            logger.warning(f"未知任务类型: {task_type}，使用默认处理")
            return self._process_default(data, **kwargs)
    
    def _process_vqa_data(self, data: List[Dict], **kwargs) -> List[Dict]:
        """处理VQA数据"""
        processed_data = []
        
        for idx, item in enumerate(data):
            try:
                processed_item = item.copy()
                
                # 验证必需字段
                required_fields = ['question', 'answer']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    logger.warning(f"样本 {idx} 缺少字段: {missing_fields}，跳过")
                    continue
                
                # 清理和标准化文本
                processed_item['question'] = self._clean_text(item.get('question', ''))
                processed_item['answer'] = self._clean_text(item.get('answer', ''))
                
                # 处理图像路径（如果需要）
                if 'image' in item or 'image_path' in item:
                    image_field = item.get('image') or item.get('image_path')
                    processed_item['image_path'] = self._normalize_image_path(image_field)
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"处理样本 {idx} 时出错: {e}，跳过")
                continue
        
        logger.info(f"VQA数据处理完成: {len(processed_data)}/{len(data)} 条成功")
        return processed_data
    
    def _process_image_captioning_data(self, data: List[Dict], **kwargs) -> List[Dict]:
        """处理Image Captioning数据"""
        processed_data = []
        
        for idx, item in enumerate(data):
            try:
                processed_item = item.copy()
                
                # 验证必需字段
                if 'caption' not in item and 'text' not in item:
                    logger.warning(f"样本 {idx} 缺少caption字段，跳过")
                    continue
                
                # 清理和标准化文本
                caption = item.get('caption') or item.get('text', '')
                processed_item['caption'] = self._clean_text(caption)
                
                # 处理图像路径
                if 'image' in item or 'image_path' in item:
                    image_field = item.get('image') or item.get('image_path')
                    processed_item['image_path'] = self._normalize_image_path(image_field)
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"处理样本 {idx} 时出错: {e}，跳过")
                continue
        
        logger.info(f"Image Captioning数据处理完成: {len(processed_data)}/{len(data)} 条成功")
        return processed_data
    
    def _process_classification_data(self, data: List[Dict], **kwargs) -> List[Dict]:
        """处理分类数据"""
        processed_data = []
        
        for idx, item in enumerate(data):
            try:
                processed_item = item.copy()
                
                # 清理文本
                if 'text' in item:
                    processed_item['text'] = self._clean_text(item['text'])
                
                # 标准化标签
                if 'label' in item:
                    processed_item['label'] = str(item['label']).strip()
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"处理样本 {idx} 时出错: {e}，跳过")
                continue
        
        logger.info(f"分类数据处理完成: {len(processed_data)}/{len(data)} 条成功")
        return processed_data
    
    def _process_default(self, data: List[Dict], **kwargs) -> List[Dict]:
        """默认处理（基本清理）"""
        processed_data = []
        
        for idx, item in enumerate(data):
            try:
                processed_item = item.copy()
                
                # 清理所有文本字段
                for key, value in processed_item.items():
                    if isinstance(value, str):
                        processed_item[key] = self._clean_text(value)
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"处理样本 {idx} 时出错: {e}，跳过")
                continue
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """清理和标准化文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符（可选）
        # text = text.strip()
        
        return text.strip()
    
    def _normalize_image_path(self, image_path: Union[str, Path]) -> str:
        """标准化图像路径"""
        if isinstance(image_path, Path):
            return str(image_path)
        return str(image_path)
    
    def _detect_task_type(self, data: List[Dict]) -> str:
        """
        根据数据字段自动检测任务类型
        
        Args:
            data: 数据列表
            
        Returns:
            检测到的任务类型
        """
        if not data:
            return 'classification'  # 默认
        
        sample = data[0]
        fields = set(sample.keys())
        
        # VQA任务：有question和answer字段
        if 'question' in fields and 'answer' in fields:
            return 'vqa'
        
        # Image Captioning任务：有caption或text字段，且有图像字段
        image_fields = {'image', 'image_path', 'image_url', 'image_base64'}
        has_image = bool(fields & image_fields)
        if ('caption' in fields or 'text' in fields) and has_image:
            return 'image_captioning'
        
        # 分类任务：有label字段
        if 'label' in fields:
            return 'classification'
        
        # Seq2Seq任务：有source和target字段
        if 'source' in fields and 'target' in fields:
            return 'seq2seq'
        
        # 默认返回classification
        return 'classification'
    
    def balance_dataset(
        self,
        data: List[Dict],
        label_field: str = 'label',
        strategy: str = 'undersample',
        random_seed: int = 42
    ) -> List[Dict]:
        """
        平衡数据集
        
        Args:
            data: 数据列表
            label_field: 标签字段名
            strategy: 平衡策略 ('undersample', 'oversample', 'none')
            random_seed: 随机种子
            
        Returns:
            平衡后的数据列表
        """
        if strategy == 'none':
            return data
        
        if label_field not in data[0]:
            logger.warning(f"数据中没有标签字段 '{label_field}'，跳过平衡")
            return data
        
        # 统计标签分布
        labels = [item[label_field] for item in data]
        label_counts = Counter(labels)
        
        logger.info(f"平衡前标签分布: {dict(label_counts)}")
        
        random.seed(random_seed)
        
        if strategy == 'undersample':
            # 下采样：保留最少数量的样本
            min_count = min(label_counts.values())
            balanced_data = []
            
            for label, count in label_counts.items():
                label_data = [item for item in data if item[label_field] == label]
                sampled = random.sample(label_data, min_count)
                balanced_data.extend(sampled)
            
            random.shuffle(balanced_data)
            logger.info(f"下采样后: {len(balanced_data)} 条数据")
            
        elif strategy == 'oversample':
            # 上采样：复制少数类样本
            max_count = max(label_counts.values())
            balanced_data = []
            
            for label, count in label_counts.items():
                label_data = [item for item in data if item[label_field] == label]
                
                if count < max_count:
                    # 需要上采样
                    num_samples = max_count - count
                    sampled = random.choices(label_data, k=num_samples)
                    label_data.extend(sampled)
                
                balanced_data.extend(label_data)
            
            random.shuffle(balanced_data)
            logger.info(f"上采样后: {len(balanced_data)} 条数据")
            
        else:
            logger.warning(f"未知的平衡策略: {strategy}，返回原始数据")
            return data
        
        # 验证平衡结果
        balanced_labels = [item[label_field] for item in balanced_data]
        balanced_counts = Counter(balanced_labels)
        logger.info(f"平衡后标签分布: {dict(balanced_counts)}")
        
        return balanced_data
    
    @staticmethod
    def split_data(
        data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42,
        stratify_by: Optional[str] = None
    ) -> tuple:
        """
        分割数据集
        
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱
            random_seed: 随机种子
            stratify_by: 分层字段（用于分层采样）
            
        Returns:
            (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")
        
        if not data:
            return [], [], []
        
        # 打乱数据
        if shuffle:
            random.seed(random_seed)
            data = data.copy()
            random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # 分层采样（如果指定）
        if stratify_by and stratify_by in data[0]:
            train_data, val_data, test_data = DataProcessor._stratified_split(
                data, train_ratio, val_ratio, test_ratio, stratify_by, random_seed
            )
        else:
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
        
        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
        logger.info(f"  验证集: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
        logger.info(f"  测试集: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    @staticmethod
    def _stratified_split(
        data: List[Dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify_by: str,
        random_seed: int
    ) -> tuple:
        """分层分割数据集"""
        from collections import defaultdict
        
        random.seed(random_seed)
        
        # 按标签分组
        grouped_data = defaultdict(list)
        for item in data:
            label = item.get(stratify_by)
            grouped_data[label].append(item)
        
        # 打乱每组数据
        for label in grouped_data:
            random.shuffle(grouped_data[label])
        
        train_data = []
        val_data = []
        test_data = []
        
        # 对每组进行分割
        for label, group_data in grouped_data.items():
            total = len(group_data)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            train_data.extend(group_data[:train_end])
            val_data.extend(group_data[train_end:val_end])
            test_data.extend(group_data[val_end:])
        
        # 打乱最终结果
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def process_batch(
        self,
        images: List[Image.Image],
        texts: Optional[List[str]] = None,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量处理图像和文本（使用processor）
        
        Args:
            images: PIL Image列表
            texts: 文本列表（可选）
            return_tensors: 返回tensor格式 ('pt', 'np', None)
            **kwargs: 其他processor参数
            
        Returns:
            处理后的字典，包含pixel_values, input_ids等
        """
        if self.processor is None:
            raise RuntimeError("processor未加载，请先调用load_processor()")
        
        if texts is None:
            # 只处理图像
            return self.processor(images=images, return_tensors=return_tensors, **kwargs)
        else:
            # 处理图像和文本
            return self.processor(
                text=texts,
                images=images,
                return_tensors=return_tensors,
                **kwargs
            )
    
    def process_image(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """
        处理单张图像
        
        Args:
            image: PIL Image
            **kwargs: 其他processor参数
            
        Returns:
            处理后的字典
        """
        if self.image_processor is None:
            raise RuntimeError("image_processor未加载")
        
        return self.image_processor(image, return_tensors="pt", **kwargs)
    
    def process_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        处理文本
        
        Args:
            text: 文本字符串或列表
            **kwargs: 其他tokenizer参数
            
        Returns:
            处理后的字典（包含input_ids, attention_mask等）
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer未加载")
        
        return self.tokenizer(text, return_tensors="pt", **kwargs)
    
    def get_processor_info(self) -> Dict[str, Any]:
        """获取processor信息"""
        return {
            'processor_type': self.processor_type,
            'processor_name': self.processor_name,
            'has_processor': self.processor is not None,
            'has_image_processor': self.image_processor is not None,
            'has_tokenizer': self.tokenizer is not None,
            'processor_class': type(self.processor).__name__ if self.processor else None,
            'image_processor_class': type(self.image_processor).__name__ if self.image_processor else None,
            'tokenizer_class': type(self.tokenizer).__name__ if self.tokenizer else None
        }


# 便捷函数：创建processor
def create_processor(
    processor_name: str,
    processor_type: Optional[str] = None,
    config: Optional[Dict] = None
) -> DataProcessor:
    """
    创建并加载processor的便捷函数
    
    Args:
        processor_name: HuggingFace模型ID或路径
        processor_type: processor类型（可选，会自动检测）
        config: 额外配置
        
    Returns:
        DataProcessor实例
    """
    processor_config = config or {}
    processor_config['processor_name'] = processor_name
    if processor_type:
        processor_config['processor_type'] = processor_type
    
    processor = DataProcessor(processor_config)
    return processor


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: 创建BLIP processor
    print("=" * 60)
    print("示例1: 创建BLIP processor")
    print("=" * 60)
    blip_processor = create_processor(
        processor_name="Salesforce/blip-vqa-base",
        processor_type="blip"
    )
    info = blip_processor.get_processor_info()
    print(f"Processor信息: {info}")
    
    # 示例2: 处理VQA数据
    print("\n" + "=" * 60)
    print("示例2: 处理VQA数据")
    print("=" * 60)
    sample_data = [
        {
            'question': 'What color is the car?',
            'answer': 'red',
            'image_path': 'path/to/image.jpg'
        },
        {
            'question': 'Is it sunny?',
            'answer': 'yes',
            'image_path': 'path/to/image2.jpg'
        }
    ]
    processed = blip_processor.process(sample_data, task_type='vqa')
    print(f"处理了 {len(processed)} 条数据")
    
    # 示例3: 数据集分割
    print("\n" + "=" * 60)
    print("示例3: 数据集分割")
    print("=" * 60)
    train, val, test = DataProcessor.split_data(
        sample_data * 10,  # 扩展到20条
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"训练集: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)}")
    
    print("\nDataProcessor模块加载完成 - 支持BLIP、CLIP等多种processor")


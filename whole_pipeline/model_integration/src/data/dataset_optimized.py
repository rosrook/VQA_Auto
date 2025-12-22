"""
内存优化的PyTorch Dataset类
针对大规模VQA和Image Captioning数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Optional, Union, Callable, Iterator
import logging
from PIL import Image
from pathlib import Path
import numpy as np
import json
import pickle
from functools import lru_cache
import gc
import os
import threading

logger = logging.getLogger(__name__)

# 调试选项：设置环境变量 DATA_DEBUG_SAVE_FIRST_N 保存前 N 张原始图像
DEBUG_SAVE_REMAINING = int(os.environ.get("DATA_DEBUG_SAVE_FIRST_N", "0") or 0)
DEBUG_SAVE_DIR = Path(os.environ.get("DATA_DEBUG_SAVE_DIR", "debug_images"))
_DEBUG_LOCK = threading.Lock()


def _save_debug_image(image: Image.Image, prefix: str, idx: int):
    """可选：保存前 N 张原始图像以便检查解码是否正确"""
    global DEBUG_SAVE_REMAINING
    if DEBUG_SAVE_REMAINING <= 0:
        return
    with _DEBUG_LOCK:
        if DEBUG_SAVE_REMAINING <= 0:
            return
        try:
            DEBUG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            filename = DEBUG_SAVE_DIR / f"{prefix}_{idx}.jpg"
            image.save(filename, format="JPEG")
            DEBUG_SAVE_REMAINING -= 1
            logger.info(f"[DEBUG] 保存调试图像: {filename}")
        except Exception as e:
            logger.warning(f"[DEBUG] 保存调试图像失败: {e}")


class ImageProcessor:
    """优化的图像处理类，支持缓存"""
    
    _cache = {}
    _cache_size = 0
    _max_cache_size = 1000  # 最多缓存1000张图像
    
    @classmethod
    def set_cache_size(cls, size: int):
        """设置缓存大小"""
        cls._max_cache_size = size
        logger.info(f"图像缓存大小设置为: {size}")
    
    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._cache.clear()
        cls._cache_size = 0
        gc.collect()
        logger.info("图像缓存已清空")
    
    @classmethod
    @lru_cache(maxsize=128)
    def _get_image_key(cls, image_path: str) -> str:
        """生成图像缓存key"""
        return str(Path(image_path).resolve())
    
    @classmethod
    def load_image(
        cls, 
        image_input: Union[str, Path, Image.Image, bytes],
        use_cache: bool = False
    ) -> Image.Image:
        """
        加载图像，支持缓存
        
        Args:
            image_input: 图像输入
            use_cache: 是否使用缓存
        """
        # 如果已经是PIL Image，直接返回
        if isinstance(image_input, Image.Image):
            return image_input
        
        # 生成缓存key
        if use_cache and isinstance(image_input, (str, Path)):
            cache_key = cls._get_image_key(str(image_input))
            if cache_key in cls._cache:
                return cls._cache[cache_key].copy()
        
        # 加载图像
        image = cls._load_image_impl(image_input)
        
        # 缓存图像（如果启用且未超过限制）
        if use_cache and isinstance(image_input, (str, Path)):
            if cls._cache_size < cls._max_cache_size:
                cls._cache[cache_key] = image.copy()
                cls._cache_size += 1
        
        return image
    
    @staticmethod
    def _is_base64_string(s: str) -> bool:
        """判断字符串是否为base64编码"""
        import base64
        # 检查是否是data URI格式
        if s.startswith('data:image'):
            return True
        
        # 检查是否是纯base64字符串
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        s_clean = s.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        if len(s_clean) > 20 and all(c in base64_chars for c in s_clean):
            try:
                base64.b64decode(s_clean, validate=True)
                return True
            except Exception:
                pass
        
        return False
    
    @staticmethod
    def _decode_base64(base64_input: str) -> bytes:
        """解码base64字符串"""
        import base64
        base64_input = base64_input.strip()
        
        # 处理data URI格式
        if base64_input.startswith('data:image'):
            try:
                if ',' in base64_input:
                    base64_str = base64_input.split(',', 1)[1]
                else:
                    raise ValueError("data URI格式无效：缺少逗号分隔符")
                image_bytes = base64.b64decode(base64_str, validate=True)
                return image_bytes
            except Exception as e:
                error_msg = f"解码data URI格式的base64失败: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        else:
            try:
                base64_str = base64_input.replace('\n', '').replace('\r', '').replace(' ', '')
                image_bytes = base64.b64decode(base64_str, validate=True)
                return image_bytes
            except Exception as e:
                error_msg = f"解码base64字符串失败: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
    
    @staticmethod
    def _load_image_impl(image_input: Union[str, Path, bytes]) -> Image.Image:
        """
        实际的图像加载逻辑
        
        支持的格式：
        1. PIL Image对象
        2. 本地文件路径（相对或绝对路径）
        3. URL（http:// 或 https://）
        4. Base64编码字符串（data URI格式或纯base64）
        5. bytes对象
        """
        import base64
        import requests
        from io import BytesIO
        
        # bytes对象
        if isinstance(image_input, bytes):
            try:
                return Image.open(BytesIO(image_input)).convert('RGB')
            except Exception as e:
                error_msg = f"无法从bytes数据加载图像: {e}"
                logger.error(error_msg)
                raise IOError(error_msg) from e
        
        # 字符串或Path对象
        elif isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            
            # URL
            if image_path.startswith(('http://', 'https://')):
                try:
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    error_msg = f"从URL加载图像失败 ({image_path[:50]}...): {e}"
                    logger.error(error_msg)
                    raise IOError(error_msg) from e
            
            # Base64编码
            elif ImageProcessor._is_base64_string(image_path):
                try:
                    image_bytes = ImageProcessor._decode_base64(image_path)
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                    return image
                except Exception as e:
                    error_msg = f"从Base64加载图像失败: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
            
            # 本地文件路径
            else:
                path = Path(image_path)
                if not path.exists():
                    error_msg = (
                        f"图像文件不存在: {image_path}\n"
                        f"提示：如果这是base64编码，请确保格式正确：\n"
                        f"  - data URI格式: data:image/jpeg;base64,/9j/4AAQ...\n"
                        f"  - 或纯base64字符串（长度>20且只包含base64字符）"
                    )
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                try:
                    return Image.open(path).convert('RGB')
                except Exception as e:
                    error_msg = f"无法打开图像文件 ({image_path}): {e}"
                    logger.error(error_msg)
                    raise IOError(error_msg) from e
        
        else:
            error_msg = (
                f"不支持的图像输入类型: {type(image_input)}\n"
                f"支持的类型: PIL.Image.Image, str, Path, bytes\n"
                f"当前输入类型: {type(image_input).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


class LazyLoadVQADataset(Dataset):
    """
    懒加载VQA数据集 - 适合大规模数据
    只在需要时才加载图像，节省内存
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        image_processor: Optional[Callable] = None,
        max_length: int = 512,
        image_field: str = 'image',
        question_field: str = 'question',
        answer_field: str = 'answer',
        image_root: Optional[str] = None,
        cache_images: bool = False,
        preload_images: bool = False,
        image_transform: Optional[Callable] = None
    ):
        """
        初始化懒加载VQA数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            image_processor: 图像处理器
            max_length: 最大文本长度
            image_field: 图像字段名
            question_field: 问题字段名
            answer_field: 答案字段名
            image_root: 图像根目录
            cache_images: 是否缓存已加载的图像
            preload_images: 是否预加载所有图像（不推荐大数据集）
            image_transform: 自定义图像转换函数
        """
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_field = image_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.image_root = Path(image_root) if image_root else None
        self.cache_images = cache_images
        self.image_transform = image_transform
        
        # 设置图像缓存
        if cache_images:
            ImageProcessor.set_cache_size(min(1000, len(data)))
        
        logger.info(f"初始化LazyLoadVQADataset，样本数: {len(self.data)}")
        logger.info(f"图像缓存: {'启用' if cache_images else '禁用'}")
        
        # 预加载图像（仅用于小数据集）
        self._preloaded_images = {}
        if preload_images:
            self._preload_all_images()
        
        # 预编码文本（可选优化）
        self._encoded_cache = {}
    
    def _preload_all_images(self):
        """预加载所有图像到内存（仅用于小数据集）"""
        logger.warning("预加载所有图像，这会占用大量内存！")
        from tqdm import tqdm
        
        for idx in tqdm(range(len(self.data)), desc="预加载图像"):
            try:
                image_path = self._get_image_path(idx)
                self._preloaded_images[idx] = ImageProcessor.load_image(
                    image_path, 
                    use_cache=False
                )
            except Exception as e:
                logger.error(f"预加载图像失败 (idx={idx}): {e}")
    
    def _get_image_path(self, idx: int) -> str:
        """获取图像路径"""
        item = self.data[idx]
        image_path = item[self.image_field]
        
        # 检查是否需要拼接image_root
        # 不拼接的情况：URL、绝对路径、base64编码
        image_path_str = str(image_path)
        is_url = image_path_str.startswith(('http://', 'https://'))
        is_absolute_path = image_path_str.startswith('/') or (len(image_path_str) > 1 and image_path_str[1] == ':')  # Windows绝对路径
        is_base64 = ImageProcessor._is_base64_string(image_path_str)
        
        if self.image_root and not (is_url or is_absolute_path or is_base64):
            # 相对路径，需要拼接image_root
            image_path = self.image_root / image_path
        
        return str(image_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """懒加载方式获取数据"""
        item = self.data[idx]
        
        # 1. 加载图像（懒加载）
        try:
            if idx in self._preloaded_images:
                image = self._preloaded_images[idx]
            else:
                image_path = self._get_image_path(idx)
                image = ImageProcessor.load_image(image_path, use_cache=self.cache_images)
            
            # 应用自定义transform
            if self.image_transform:
                image = self.image_transform(image)
        
        except Exception as e:
            logger.warning(f"加载图像失败 (idx={idx}): {e}，使用空白图像")
            image = Image.new('RGB', (224, 224), color='white')
        
        # 调试：保存前N张原始图像
        _save_debug_image(image, prefix="lazy_vqa_raw", idx=idx)
        
        # 2. 处理图像
        if self.image_processor is not None:
            processed = self.image_processor(images=image, return_tensors="pt")
            # BatchFeature是dict-like，支持字典式访问，但为了安全先检查
            if hasattr(processed, 'pixel_values') or 'pixel_values' in processed:
                pixel_values = processed['pixel_values'].squeeze(0)
            else:
                raise ValueError(f"无法从processor输出解析pixel_values: type={type(processed)}, keys={list(processed.keys()) if hasattr(processed, 'keys') else 'N/A'}")
        else:
            import torchvision.transforms as transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = default_transform(image)
        
        # 3. 处理文本（使用缓存）
        if idx in self._encoded_cache:
            question_encoding, answer_encoding = self._encoded_cache[idx]
        else:
            question = item.get(self.question_field, "")
            if not isinstance(question, str):
                question = str(question) if question is not None else ""
            question = question.strip() or "What is in the image?"
            
            answer = item.get(self.answer_field, "")
            if not isinstance(answer, str):
                answer = str(answer) if answer is not None else ""
            answer = answer.strip() or "unknown"
            
            question_encoding = self.tokenizer(
                question,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            answer_encoding = self.tokenizer(
                answer,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 验证和修复token IDs
            question_encoding = self._validate_and_fix_token_ids(question_encoding, "question")
            answer_encoding = self._validate_and_fix_token_ids(answer_encoding, "answer")
            
            # 创建labels（mask padding tokens）
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                answer_labels = answer_encoding['input_ids'].clone()
                answer_labels[answer_labels == pad_id] = -100
                answer_encoding['labels'] = answer_labels
            else:
                answer_encoding['labels'] = answer_encoding['input_ids'].clone()
        
        return {
            'pixel_values': pixel_values,
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'labels': answer_encoding.get('labels', answer_encoding['input_ids']).squeeze(0),
        }
    
    def _validate_and_fix_token_ids(self, encoding: Dict[str, torch.Tensor], field_name: str) -> Dict[str, torch.Tensor]:
        """
        验证并修复token IDs，确保它们在有效范围内
        
        Args:
            encoding: tokenizer返回的编码字典
            field_name: 字段名称（用于日志）
            
        Returns:
            修复后的编码字典
        """
        input_ids = encoding['input_ids']
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
        
        if vocab_size is not None:
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.warning(
                    f"发现超出词汇表大小的token ID ({field_name}): {max_token_id} >= {vocab_size}"
                )
                # 将超出范围的token替换为unk_token_id或pad_token_id
                unk_id = getattr(self.tokenizer, 'unk_token_id', None) or self.tokenizer.pad_token_id or 0
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                # 如果clamp后还有问题，用unk_id替换
                if input_ids.max().item() >= vocab_size:
                    input_ids[input_ids >= vocab_size] = unk_id
                encoding['input_ids'] = input_ids
        
        return encoding
    
    def clear_cache(self):
        """清空所有缓存"""
        self._encoded_cache.clear()
        ImageProcessor.clear_cache()
        gc.collect()


class StreamingVQADataset(IterableDataset):
    """
    流式VQA数据集 - 适合超大规模数据
    不将所有数据加载到内存，而是逐行读取
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        image_processor: Optional[Callable] = None,
        max_length: int = 512,
        image_field: str = 'image',
        question_field: str = 'question',
        answer_field: str = 'answer',
        image_root: Optional[str] = None,
        file_format: str = 'jsonl',
        skip_errors: bool = True
    ):
        """
        初始化流式数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            image_processor: 图像处理器
            max_length: 最大文本长度
            image_field: 图像字段名
            question_field: 问题字段名
            answer_field: 答案字段名
            image_root: 图像根目录
            file_format: 文件格式 ('jsonl', 'parquet')
            skip_errors: 是否跳过错误样本
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_field = image_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.image_root = Path(image_root) if image_root else None
        self.file_format = file_format
        self.skip_errors = skip_errors
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        logger.info(f"初始化StreamingVQADataset: {data_path}")
        logger.info(f"流式加载，不占用大量内存")
    
    def _parse_jsonl(self) -> Iterator[Dict]:
        """解析JSONL文件"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    if not self.skip_errors:
                        raise
                    logger.warning(f"跳过无效JSON (行{line_num}): {e}")
    
    def _parse_parquet(self) -> Iterator[Dict]:
        """解析Parquet文件（分批读取）"""
        import pandas as pd
        
        # 使用迭代器读取，每次读取1000行
        for chunk in pd.read_parquet(self.data_path, chunksize=1000):
            for _, row in chunk.iterrows():
                yield row.to_dict()
    
    def _get_data_iterator(self) -> Iterator[Dict]:
        """获取数据迭代器"""
        if self.file_format == 'jsonl':
            return self._parse_jsonl()
        elif self.file_format == 'parquet':
            return self._parse_parquet()
        else:
            raise ValueError(f"不支持的文件格式: {self.file_format}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """流式迭代数据"""
        for item in self._get_data_iterator():
            try:
                # 获取图像路径
                image_path = item[self.image_field]
                
                # 检查是否需要拼接image_root
                # 不拼接的情况：URL、绝对路径、base64编码
                image_path_str = str(image_path)
                is_url = image_path_str.startswith(('http://', 'https://'))
                is_absolute_path = image_path_str.startswith('/') or (len(image_path_str) > 1 and image_path_str[1] == ':')  # Windows绝对路径
                is_base64 = ImageProcessor._is_base64_string(image_path_str)
                
                if self.image_root and not (is_url or is_absolute_path or is_base64):
                    # 相对路径，需要拼接image_root
                    image_path = self.image_root / image_path
                
                # 加载图像
                image = ImageProcessor.load_image(image_path, use_cache=False)
                
                # 调试：保存前N张原始图像
                _save_debug_image(image, prefix="stream_vqa_raw", idx=0)
                
                # 处理图像
                if self.image_processor is not None:
                    processed = self.image_processor(images=image, return_tensors="pt")
                    # BatchFeature是dict-like，支持字典式访问，但为了安全先检查
                    if hasattr(processed, 'pixel_values') or 'pixel_values' in processed:
                        pixel_values = processed['pixel_values'].squeeze(0)
                    else:
                        raise ValueError(f"无法从processor输出解析pixel_values: type={type(processed)}, keys={list(processed.keys()) if hasattr(processed, 'keys') else 'N/A'}")
                else:
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    pixel_values = transform(image)
                
                # 处理文本
                question = item.get(self.question_field, "")
                if not isinstance(question, str):
                    question = str(question) if question is not None else ""
                question = question.strip() or "What is in the image?"
                
                answer = item.get(self.answer_field, "")
                if not isinstance(answer, str):
                    answer = str(answer) if answer is not None else ""
                answer = answer.strip() or "unknown"
                
                question_encoding = self.tokenizer(
                    question,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                answer_encoding = self.tokenizer(
                    answer,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 验证和修复token IDs
                question_encoding = self._validate_and_fix_token_ids(question_encoding, "question")
                answer_encoding = self._validate_and_fix_token_ids(answer_encoding, "answer")
                
                # 创建labels（mask padding tokens）
                pad_id = self.tokenizer.pad_token_id
                if pad_id is not None:
                    answer_labels = answer_encoding['input_ids'].clone()
                    answer_labels[answer_labels == pad_id] = -100
                    answer_encoding['labels'] = answer_labels
                else:
                    answer_encoding['labels'] = answer_encoding['input_ids'].clone()
                
                yield {
                    'pixel_values': pixel_values,
                    'input_ids': question_encoding['input_ids'].squeeze(0),
                    'attention_mask': question_encoding['attention_mask'].squeeze(0),
                    'labels': answer_encoding.get('labels', answer_encoding['input_ids']).squeeze(0),
                }
            
            except Exception as e:
                if not self.skip_errors:
                    raise
                logger.warning(f"跳过错误样本: {e}")
                continue
    
    def _validate_and_fix_token_ids(self, encoding: Dict[str, torch.Tensor], field_name: str) -> Dict[str, torch.Tensor]:
        """
        验证并修复token IDs，确保它们在有效范围内
        
        Args:
            encoding: tokenizer返回的编码字典
            field_name: 字段名称（用于日志）
            
        Returns:
            修复后的编码字典
        """
        input_ids = encoding['input_ids']
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
        
        if vocab_size is not None:
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.warning(
                    f"发现超出词汇表大小的token ID ({field_name}): {max_token_id} >= {vocab_size}"
                )
                # 将超出范围的token替换为unk_token_id或pad_token_id
                unk_id = getattr(self.tokenizer, 'unk_token_id', None) or self.tokenizer.pad_token_id or 0
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                # 如果clamp后还有问题，用unk_id替换
                if input_ids.max().item() >= vocab_size:
                    input_ids[input_ids >= vocab_size] = unk_id
                encoding['input_ids'] = input_ids
        
        return encoding


class MemoryMappedVQADataset(Dataset):
    """
    内存映射VQA数据集 - 适合超大规模数据
    使用numpy memmap技术，不占用内存
    需要预处理数据为二进制格式
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        image_processor: Optional[Callable] = None,
    ):
        """
        初始化内存映射数据集
        
        Args:
            data_dir: 预处理数据目录
            tokenizer: 分词器
            image_processor: 图像处理器
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # 加载元数据
        metadata_path = self.data_dir / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.length = self.metadata['length']
        
        # 创建内存映射
        self.images_mmap = np.memmap(
            self.data_dir / 'images.mmap',
            dtype='uint8',
            mode='r',
            shape=(self.length, 3, 224, 224)
        )
        
        self.questions_mmap = np.memmap(
            self.data_dir / 'questions.mmap',
            dtype='int64',
            mode='r',
            shape=(self.length, self.metadata['max_question_length'])
        )
        
        self.answers_mmap = np.memmap(
            self.data_dir / 'answers.mmap',
            dtype='int64',
            mode='r',
            shape=(self.length, self.metadata['max_answer_length'])
        )
        
        logger.info(f"初始化MemoryMappedVQADataset，样本数: {self.length}")
        logger.info(f"使用内存映射，几乎不占用内存")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 从内存映射读取数据（不会加载到内存）
        image = torch.from_numpy(self.images_mmap[idx].copy()).float() / 255.0
        question_ids = torch.from_numpy(self.questions_mmap[idx].copy())
        answer_ids = torch.from_numpy(self.answers_mmap[idx].copy())
        
        # 创建attention mask
        question_mask = (question_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'pixel_values': image,
            'input_ids': question_ids,
            'attention_mask': question_mask,
            'labels': answer_ids,
        }


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """
    创建优化的DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        prefetch_factor: 预取因子
        persistent_workers: 是否保持工作进程
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )


# 示例：预处理数据为内存映射格式
def preprocess_to_memmap(
    data_path: str,
    output_dir: str,
    tokenizer,
    image_processor,
    max_question_length: int = 128,
    max_answer_length: int = 32,
    image_size: tuple = (224, 224)
):
    """
    将数据预处理为内存映射格式
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
        tokenizer: 分词器
        image_processor: 图像处理器
        max_question_length: 最大问题长度
        max_answer_length: 最大答案长度
        image_size: 图像尺寸
    """
    from data.data_loader import DataLoader as CustomDataLoader
    from tqdm import tqdm
    
    logger.info("开始预处理数据...")
    
    # 加载数据
    loader = CustomDataLoader(data_path)
    data = loader.load()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    length = len(data)
    
    # 创建内存映射文件
    images_mmap = np.memmap(
        output_dir / 'images.mmap',
        dtype='uint8',
        mode='w+',
        shape=(length, 3, *image_size)
    )
    
    questions_mmap = np.memmap(
        output_dir / 'questions.mmap',
        dtype='int64',
        mode='w+',
        shape=(length, max_question_length)
    )
    
    answers_mmap = np.memmap(
        output_dir / 'answers.mmap',
        dtype='int64',
        mode='w+',
        shape=(length, max_answer_length)
    )
    
    # 处理数据
    for idx, item in enumerate(tqdm(data, desc="预处理")):
        # 处理图像
        image = ImageProcessor.load_image(item['image'])
        image = image.resize(image_size)
        image_array = np.array(image).transpose(2, 0, 1)
        images_mmap[idx] = image_array
        
        # 处理文本
        question_encoding = tokenizer(
            item['question'],
            max_length=max_question_length,
            padding='max_length',
            truncation=True
        )
        questions_mmap[idx] = question_encoding['input_ids']
        
        answer_encoding = tokenizer(
            item['answer'],
            max_length=max_answer_length,
            padding='max_length',
            truncation=True
        )
        answers_mmap[idx] = answer_encoding['input_ids']
    
    # 保存元数据
    metadata = {
        'length': length,
        'max_question_length': max_question_length,
        'max_answer_length': max_answer_length,
        'image_size': image_size
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"预处理完成，数据保存至: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("内存优化的Dataset模块加载完成")
    print("\n可用的数据集类:")
    print("1. LazyLoadVQADataset - 懒加载，适合大数据集")
    print("2. StreamingVQADataset - 流式加载，适合超大数据集")
    print("3. MemoryMappedVQADataset - 内存映射，适合超大数据集（需要预处理）")
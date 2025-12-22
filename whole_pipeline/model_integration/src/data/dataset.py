"""
PyTorch Dataset类
专门针对VQA和Image Captioning任务，同时保留其他任务的支持
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Optional, Union, Callable, Any
import logging
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
import requests
import warnings
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
    """图像处理辅助类"""
    
    @staticmethod
    def _is_base64_string(s: str) -> bool:
        """
        判断字符串是否为base64编码
        
        Args:
            s: 待检查的字符串
            
        Returns:
            是否为base64编码
        """
        # 检查是否是data URI格式
        if s.startswith('data:image'):
            return True
        
        # 检查是否是纯base64字符串（尝试解码）
        # Base64字符串通常只包含A-Z, a-z, 0-9, +, /, = 字符
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        s_clean = s.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # 如果字符串长度合理且只包含base64字符，可能是base64
        if len(s_clean) > 20 and all(c in base64_chars for c in s_clean):
            # 尝试解码验证
            try:
                base64.b64decode(s_clean, validate=True)
                return True
            except Exception:
                pass
        
        return False
    
    @staticmethod
    def _decode_base64(base64_input: str) -> bytes:
        """
        解码base64字符串
        
        Args:
            base64_input: base64编码的字符串
            
        Returns:
            解码后的字节数据
            
        Raises:
            ValueError: 如果base64格式无效
        """
        base64_input = base64_input.strip()
        
        # 处理data URI格式: data:image/jpeg;base64,/9j/4AAQ...
        if base64_input.startswith('data:image'):
            try:
                # 提取base64部分（逗号后面的部分）
                if ',' in base64_input:
                    base64_str = base64_input.split(',', 1)[1]
                else:
                    raise ValueError("data URI格式无效：缺少逗号分隔符")
                
                # 解码
                image_bytes = base64.b64decode(base64_str, validate=True)
                return image_bytes
            except Exception as e:
                error_msg = f"解码data URI格式的base64失败: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        # 处理纯base64字符串
        else:
            try:
                # 移除可能的空白字符
                base64_str = base64_input.replace('\n', '').replace('\r', '').replace(' ', '')
                image_bytes = base64.b64decode(base64_str, validate=True)
                return image_bytes
            except Exception as e:
                error_msg = f"解码base64字符串失败: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
    
    @staticmethod
    def load_image(image_input: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """
        加载图像，支持多种输入格式
        
        支持的格式：
        1. PIL Image对象
        2. 本地文件路径（相对或绝对路径）
        3. URL（http:// 或 https://）
        4. Base64编码字符串：
           - data URI格式: data:image/jpeg;base64,/9j/4AAQ...
           - 纯base64字符串: /9j/4AAQSkZJRg...
        5. bytes对象（图像字节数据）
        
        Args:
            image_input: 图像输入，可以是路径、PIL Image、base64字符串或字节
            
        Returns:
            PIL Image对象
            
        Raises:
            ValueError: 如果输入格式不支持或无法解析
            FileNotFoundError: 如果文件路径不存在
            IOError: 如果图像数据无效或无法解码
        """
        # 1. PIL Image对象
        if isinstance(image_input, Image.Image):
            return image_input
        
        # 2. bytes对象
        elif isinstance(image_input, bytes):
            try:
                return Image.open(BytesIO(image_input)).convert('RGB')
            except Exception as e:
                error_msg = f"无法从bytes数据加载图像: {e}"
                logger.error(error_msg)
                raise IOError(error_msg) from e
        
        # 3. 字符串或Path对象
        elif isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            
            # 3.1 URL
            if image_path.startswith(('http://', 'https://')):
                try:
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    error_msg = f"从URL加载图像失败 ({image_path[:50]}...): {e}"
                    logger.error(error_msg)
                    raise IOError(error_msg) from e
            
            # 3.2 Base64编码（data URI格式或纯base64）
            elif ImageProcessor._is_base64_string(image_path):
                try:
                    image_bytes = ImageProcessor._decode_base64(image_path)
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                    return image
                except Exception as e:
                    error_msg = f"从Base64加载图像失败: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
            
            # 3.3 本地文件路径
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
        
        # 4. 不支持的类型
        else:
            error_msg = (
                f"不支持的图像输入类型: {type(image_input)}\n"
                f"支持的类型: PIL.Image.Image, str, Path, bytes\n"
                f"当前输入类型: {type(image_input).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def preprocess_image(
        image: Image.Image,
        image_processor: Optional[Callable] = None,
        size: Optional[tuple] = None
    ) -> Union[Image.Image, torch.Tensor]:
        """
        预处理图像
        
        Args:
            image: PIL Image对象
            image_processor: 图像处理器（如来自transformers的processor）
            size: 目标尺寸 (width, height)
            
        Returns:
            处理后的图像
        """
        if image_processor is not None:
            # 使用transformers的processor
            return image_processor(images=image, return_tensors="pt")
        
        if size is not None:
            image = image.resize(size, Image.LANCZOS)
        
        return image


class VQADataset(Dataset):
    """视觉问答(VQA)数据集"""
    
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
        return_raw_image: bool = False
    ):
        """
        初始化VQA数据集
        
        Args:
            data: 数据列表，每个元素包含image, question, answer
            tokenizer: 文本分词器
            image_processor: 图像处理器（可选，如CLIP processor）
            max_length: 最大文本长度
            image_field: 图像字段名
            question_field: 问题字段名
            answer_field: 答案字段名
            image_root: 图像根目录（如果image_field是相对路径）
            return_raw_image: 是否返回原始PIL图像
        """
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_field = image_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.image_root = Path(image_root) if image_root else None
        self.return_raw_image = return_raw_image
        self._default_image_transform = None
        
        logger.info(f"初始化VQA数据集，样本数: {len(self.data)}")
        
        # 验证数据
        self._validate_data()
        # 预构建默认图像transform（仅当未提供image_processor时使用）
        self._build_default_transform()
    
    def _validate_data(self):
        """验证数据格式"""
        if not self.data:
            raise ValueError("数据为空")
        
        required_fields = [self.image_field, self.question_field, self.answer_field]
        missing_fields = [f for f in required_fields if f not in self.data[0]]
        if missing_fields:
            raise ValueError(f"数据缺少必需字段: {missing_fields}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 1. 加载图像
        image_input = item[self.image_field]
        
        # 检查是否需要拼接image_root
        # 不拼接的情况：URL、绝对路径、base64编码
        image_input_str = str(image_input)
        is_url = image_input_str.startswith(('http://', 'https://'))
        is_absolute_path = image_input_str.startswith('/') or (len(image_input_str) > 1 and image_input_str[1] == ':')  # Windows绝对路径
        is_base64 = ImageProcessor._is_base64_string(image_input_str)
        
        if self.image_root and not (is_url or is_absolute_path or is_base64):
            # 相对路径，需要拼接image_root
            image_input = self.image_root / image_input
        
        try:
            image = ImageProcessor.load_image(image_input)
        except Exception as e:
            error_msg = (
                f"加载图像失败 (idx={idx}, image_field={self.image_field}): {e}\n"
                f"图像输入类型: {type(image_input).__name__}\n"
                f"图像输入值（前100字符）: {str(image_input)[:100]}"
            )
            logger.error(error_msg)
            # 创建空白图像作为fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # 调试：保存前N张原始图像
        _save_debug_image(image, prefix="vqa_raw", idx=idx)
        
        # 2. 处理图像
        pixel_values = self._process_image(image)
        
        # 3. 处理文本（问题）
        question = item.get(self.question_field, "")
        if not isinstance(question, str):
            question = str(question) if question is not None else ""
        question = question.strip()
        if not question:
            logger.warning(f"样本 {idx} 的问题为空，使用默认问题")
            question = "What is in the image?"
        
        question_encoding = self._tokenize(question)
        
        # 4. 处理答案
        answer = item.get(self.answer_field, "")
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""
        answer = answer.strip()
        if not answer:
            logger.warning(f"样本 {idx} 的答案为空，使用默认答案")
            answer = "unknown"
        
        answer_encoding = self._tokenize(answer, mask_labels=True)
        
        # 验证tensor shapes
        try:
            result = {
                'pixel_values': pixel_values,
                'input_ids': question_encoding['input_ids'].squeeze(0),
                'attention_mask': question_encoding['attention_mask'].squeeze(0),
                'labels': answer_encoding.get('labels', answer_encoding['input_ids']).squeeze(0),
            }
            
            # 最终验证：确保所有tensor的shape正确
            if result['input_ids'].dim() != 1:
                raise ValueError(f"input_ids维度错误: {result['input_ids'].shape}, 期望1D")
            if result['labels'].dim() != 1:
                raise ValueError(f"labels维度错误: {result['labels'].shape}, 期望1D")
            if result['input_ids'].size(0) != result['labels'].size(0):
                raise ValueError(f"input_ids和labels长度不匹配: {result['input_ids'].size(0)} vs {result['labels'].size(0)}")
            
            # 验证token IDs范围
            self._validate_batch_item(result, idx)
            
        except Exception as e:
            logger.error(f"样本 {idx} 处理失败: {e}")
            logger.error(f"  question: {question[:50]}...")
            logger.error(f"  answer: {answer[:50]}...")
            logger.error(f"  question_encoding keys: {list(question_encoding.keys())}")
            logger.error(f"  answer_encoding keys: {list(answer_encoding.keys())}")
            raise
        
        # 可选：返回原始图像和文本
        if self.return_raw_image:
            result['raw_image'] = image
            result['raw_question'] = question
            result['raw_answer'] = answer
        
        return result

    def _build_default_transform(self):
        """仅在未提供image_processor时构建默认的图像transform"""
        if self.image_processor is None:
            try:
                import torchvision.transforms as transforms
                self._default_image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                warnings.warn(f"未安装torchvision，使用原始图像张量: {e}")
                self._default_image_transform = None

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """统一的图像处理，兼容HF processor与默认transform"""
        if self.image_processor is not None:
            processed = self.image_processor(images=image, return_tensors="pt")
            # 兼容processor返回dict/BatchFeature
            # BatchFeature是dict-like，但isinstance(processed, dict)在某些版本可能失败
            # 直接检查是否有pixel_values键更安全
            if hasattr(processed, 'pixel_values') or 'pixel_values' in processed:
                return processed['pixel_values'].squeeze(0)
            # 如果processor直接返回tensor
            if isinstance(processed, torch.Tensor):
                return processed
            raise ValueError(f"无法从processor输出解析pixel_values: type={type(processed)}, keys={list(processed.keys()) if hasattr(processed, 'keys') else 'N/A'}")
        else:
            if self._default_image_transform:
                return self._default_image_transform(image)
            # 最简 fallback：转为tensor [H,W,3] -> [3,H,W], 0-1
            return torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float())

    def _tokenize(self, text: str, mask_labels: bool = False) -> Dict[str, torch.Tensor]:
        """
        统一的文本tokenize，并可选地对padding部分mask为-100
        
        Args:
            text: 要tokenize的文本
            mask_labels: 是否将padding token mask为-100（用于labels）
            
        Returns:
            包含input_ids、attention_mask和labels的字典
        """
        # 验证输入
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()
        if not text:
            text = " "  # 空文本用空格代替，避免tokenize失败
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding.get('attention_mask', None)
        
        # 验证token IDs是否在有效范围内
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
        if vocab_size is not None:
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.warning(
                    f"发现超出词汇表大小的token ID: {max_token_id} >= {vocab_size}, "
                    f"文本: {text[:50]}..."
                )
                # 将超出范围的token替换为unk_token_id或pad_token_id
                unk_id = getattr(self.tokenizer, 'unk_token_id', None) or self.tokenizer.pad_token_id or 0
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                # 如果clamp后还有问题，用unk_id替换
                if input_ids.max().item() >= vocab_size:
                    input_ids[input_ids >= vocab_size] = unk_id
                encoding['input_ids'] = input_ids
        
        # 确保attention_mask存在且正确
        if attention_mask is None:
            # 如果没有attention_mask，根据input_ids创建
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                attention_mask = (input_ids != pad_id).long()
            else:
                # 如果没有pad_token_id，假设所有非0位置都是有效token
                attention_mask = (input_ids != 0).long()
            encoding['attention_mask'] = attention_mask
        else:
            # 验证attention_mask shape
            if attention_mask.shape != input_ids.shape:
                logger.warning(
                    f"attention_mask shape {attention_mask.shape} 与 input_ids shape {input_ids.shape} 不匹配，"
                    f"重新创建attention_mask"
                )
                pad_id = self.tokenizer.pad_token_id
                if pad_id is not None:
                    attention_mask = (input_ids != pad_id).long()
                else:
                    attention_mask = (input_ids != 0).long()
                encoding['attention_mask'] = attention_mask
            
            # 确保attention_mask值是0或1
            unique_values = torch.unique(attention_mask)
            invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
            if len(invalid_values) > 0:
                logger.warning(
                    f"attention_mask包含非法值: {invalid_values.tolist()}，clamp到[0, 1]"
                )
                attention_mask = torch.clamp(attention_mask, 0, 1).long()
                encoding['attention_mask'] = attention_mask
        
        if mask_labels:
            # 创建labels（将padding token mask为-100）
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                labels = input_ids.clone()
                labels[labels == pad_id] = -100
                encoding['labels'] = labels
            else:
                # 如果没有pad_token_id，直接使用input_ids作为labels
                encoding['labels'] = input_ids.clone()
        else:
            # 即使不mask，也返回labels（用于某些模型）
            encoding['labels'] = input_ids.clone()
        
        return encoding
    
    def _validate_batch_item(self, item: Dict[str, torch.Tensor], idx: int):
        """
        验证batch item的tensor shapes和值范围
        
        Args:
            item: 包含pixel_values, input_ids, labels等的字典
            idx: 样本索引（用于错误报告）
        """
        # 验证input_ids
        if 'input_ids' in item:
            input_ids = item['input_ids']
            # 确保是1D tensor（单个样本）
            if input_ids.dim() != 1:
                raise ValueError(f"样本 {idx} input_ids应该是1D tensor，得到shape {input_ids.shape}")
            
            vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
            if vocab_size is not None:
                max_id = input_ids.max().item()
                min_id = input_ids.min().item()
                if max_id >= vocab_size or min_id < 0:
                    logger.error(
                        f"样本 {idx} input_ids超出范围: min={min_id}, max={max_id}, vocab_size={vocab_size}"
                    )
                    raise ValueError(f"input_ids超出词汇表范围: [{min_id}, {max_id}] vs [0, {vocab_size-1}]")
        
        # 验证attention_mask
        if 'attention_mask' in item:
            attention_mask = item['attention_mask']
            if attention_mask.dim() != 1:
                raise ValueError(f"样本 {idx} attention_mask应该是1D tensor，得到shape {attention_mask.shape}")
            
            # 确保attention_mask长度与input_ids匹配
            if 'input_ids' in item:
                if attention_mask.size(0) != item['input_ids'].size(0):
                    raise ValueError(
                        f"样本 {idx} attention_mask长度 {attention_mask.size(0)} 与 "
                        f"input_ids长度 {item['input_ids'].size(0)} 不匹配"
                    )
            
            # 验证attention_mask值（应该是0或1）
            unique_values = torch.unique(attention_mask)
            invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
            if len(invalid_values) > 0:
                logger.warning(
                    f"样本 {idx} attention_mask包含非法值: {invalid_values.tolist()}，"
                    f"将clamp到[0, 1]范围"
                )
                item['attention_mask'] = torch.clamp(attention_mask, 0, 1).long()
        
        # 验证labels
        if 'labels' in item:
            labels = item['labels']
            if labels.dim() != 1:
                raise ValueError(f"样本 {idx} labels应该是1D tensor，得到shape {labels.shape}")
            
            # 确保labels长度与input_ids匹配
            if 'input_ids' in item:
                if labels.size(0) != item['input_ids'].size(0):
                    raise ValueError(
                        f"样本 {idx} labels长度 {labels.size(0)} 与 "
                        f"input_ids长度 {item['input_ids'].size(0)} 不匹配"
                    )
            
            # labels可以是-100（masked），也可以是token IDs
            valid_labels = labels[(labels != -100) & (labels >= 0)]
            if len(valid_labels) > 0:
                vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
                if vocab_size is not None:
                    max_label = valid_labels.max().item()
                    if max_label >= vocab_size:
                        logger.error(
                            f"样本 {idx} labels超出范围: max={max_label}, vocab_size={vocab_size}"
                        )
                        raise ValueError(f"labels超出词汇表范围: max={max_label} >= {vocab_size}")


class ImageCaptioningDataset(Dataset):
    """图像描述(Image Captioning)数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        image_processor: Optional[Callable] = None,
        max_length: int = 128,
        image_field: str = 'image',
        caption_field: str = 'caption',
        image_root: Optional[str] = None,
        multiple_captions: bool = False,
        return_raw_image: bool = False
    ):
        """
        初始化Image Captioning数据集
        
        Args:
            data: 数据列表，每个元素包含image和caption
            tokenizer: 文本分词器
            image_processor: 图像处理器
            max_length: caption最大长度
            image_field: 图像字段名
            caption_field: caption字段名
            image_root: 图像根目录
            multiple_captions: 是否支持多个captions（随机选择一个）
            return_raw_image: 是否返回原始PIL图像
        """
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_field = image_field
        self.caption_field = caption_field
        self.image_root = Path(image_root) if image_root else None
        self.multiple_captions = multiple_captions
        self.return_raw_image = return_raw_image
        self._default_image_transform = None
        
        logger.info(f"初始化Image Captioning数据集，样本数: {len(self.data)}")
        self._validate_data()
        self._build_default_transform()
    
    def _validate_data(self):
        """验证数据格式"""
        if not self.data:
            raise ValueError("数据为空")
        
        required_fields = [self.image_field, self.caption_field]
        missing_fields = [f for f in required_fields if f not in self.data[0]]
        if missing_fields:
            raise ValueError(f"数据缺少必需字段: {missing_fields}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 1. 加载图像
        image_input = item[self.image_field]
        
        # 检查是否需要拼接image_root
        # 不拼接的情况：URL、绝对路径、base64编码
        image_input_str = str(image_input)
        is_url = image_input_str.startswith(('http://', 'https://'))
        is_absolute_path = image_input_str.startswith('/') or (len(image_input_str) > 1 and image_input_str[1] == ':')  # Windows绝对路径
        is_base64 = ImageProcessor._is_base64_string(image_input_str)
        
        if self.image_root and not (is_url or is_absolute_path or is_base64):
            # 相对路径，需要拼接image_root
            image_input = self.image_root / image_input
        
        try:
            image = ImageProcessor.load_image(image_input)
        except Exception as e:
            error_msg = (
                f"加载图像失败 (idx={idx}, image_field={self.image_field}): {e}\n"
                f"图像输入类型: {type(image_input).__name__}\n"
                f"图像输入值（前100字符）: {str(image_input)[:100]}"
            )
            logger.error(error_msg)
            image = Image.new('RGB', (224, 224), color='white')
        
        # 调试：保存前N张原始图像
        _save_debug_image(image, prefix="caption_raw", idx=idx)
        
        # 2. 处理图像
        pixel_values = self._process_image(image)
        
        # 3. 处理caption
        caption = item[self.caption_field]
        
        # 如果有多个captions，随机选择一个
        if self.multiple_captions and isinstance(caption, list):
            import random
            caption = random.choice(caption)
        
        caption_encoding = self._tokenize(caption, mask_labels=True)
        
        result = {
            'pixel_values': pixel_values,
            'input_ids': caption_encoding['input_ids'].squeeze(0),
            'attention_mask': caption_encoding['attention_mask'].squeeze(0),
            'labels': caption_encoding['input_ids'].squeeze(0),
        }
        
        if self.return_raw_image:
            result['raw_image'] = image
            result['raw_caption'] = caption
        
        return result

    def _build_default_transform(self):
        """仅在未提供image_processor时构建默认的图像transform"""
        if self.image_processor is None:
            try:
                import torchvision.transforms as transforms
                self._default_image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                warnings.warn(f"未安装torchvision，使用原始图像张量: {e}")
                self._default_image_transform = None

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """统一的图像处理，兼容HF processor与默认transform"""
        if self.image_processor is not None:
            processed = self.image_processor(images=image, return_tensors="pt")
            # 兼容processor返回dict/BatchFeature
            # BatchFeature是dict-like，但isinstance(processed, dict)在某些版本可能失败
            # 直接检查是否有pixel_values键更安全
            if hasattr(processed, 'pixel_values') or 'pixel_values' in processed:
                return processed['pixel_values'].squeeze(0)
            if isinstance(processed, torch.Tensor):
                return processed
            raise ValueError(f"无法从processor输出解析pixel_values: type={type(processed)}, keys={list(processed.keys()) if hasattr(processed, 'keys') else 'N/A'}")
        else:
            if self._default_image_transform:
                return self._default_image_transform(image)
            return torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float())

    def _tokenize(self, text: str, mask_labels: bool = False) -> Dict[str, torch.Tensor]:
        """
        统一的文本tokenize，并可选地对padding部分mask为-100
        
        Args:
            text: 要tokenize的文本
            mask_labels: 是否将padding token mask为-100（用于labels）
            
        Returns:
            包含input_ids、attention_mask和labels的字典
        """
        # 验证输入
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()
        if not text:
            text = " "  # 空文本用空格代替，避免tokenize失败
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding.get('attention_mask', None)
        
        # 验证token IDs是否在有效范围内
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else getattr(self.tokenizer, 'vocab_size', None)
        if vocab_size is not None:
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.warning(
                    f"发现超出词汇表大小的token ID: {max_token_id} >= {vocab_size}, "
                    f"文本: {text[:50]}..."
                )
                # 将超出范围的token替换为unk_token_id或pad_token_id
                unk_id = getattr(self.tokenizer, 'unk_token_id', None) or self.tokenizer.pad_token_id or 0
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                # 如果clamp后还有问题，用unk_id替换
                if input_ids.max().item() >= vocab_size:
                    input_ids[input_ids >= vocab_size] = unk_id
                encoding['input_ids'] = input_ids
        
        # 确保attention_mask存在且正确
        if attention_mask is None:
            # 如果没有attention_mask，根据input_ids创建
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                attention_mask = (input_ids != pad_id).long()
            else:
                # 如果没有pad_token_id，假设所有非0位置都是有效token
                attention_mask = (input_ids != 0).long()
            encoding['attention_mask'] = attention_mask
        else:
            # 验证attention_mask shape
            if attention_mask.shape != input_ids.shape:
                logger.warning(
                    f"attention_mask shape {attention_mask.shape} 与 input_ids shape {input_ids.shape} 不匹配，"
                    f"重新创建attention_mask"
                )
                pad_id = self.tokenizer.pad_token_id
                if pad_id is not None:
                    attention_mask = (input_ids != pad_id).long()
                else:
                    attention_mask = (input_ids != 0).long()
                encoding['attention_mask'] = attention_mask
            
            # 确保attention_mask值是0或1
            unique_values = torch.unique(attention_mask)
            invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
            if len(invalid_values) > 0:
                logger.warning(
                    f"attention_mask包含非法值: {invalid_values.tolist()}，clamp到[0, 1]"
                )
                attention_mask = torch.clamp(attention_mask, 0, 1).long()
                encoding['attention_mask'] = attention_mask
        
        if mask_labels:
            # 创建labels（将padding token mask为-100）
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                labels = input_ids.clone()
                labels[labels == pad_id] = -100
                encoding['labels'] = labels
            else:
                # 如果没有pad_token_id，直接使用input_ids作为labels
                encoding['labels'] = input_ids.clone()
        else:
            # 即使不mask，也返回labels（用于某些模型）
            encoding['labels'] = input_ids.clone()
        
        return encoding


# ============== 保留其他任务的Dataset类（向后兼容） ==============

class BaseTextDataset(Dataset):
    """基础文本数据集类"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_field: str = 'text',
        label_field: str = 'label'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field
        logger.info(f"初始化数据集，样本数: {len(self.data)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get(self.text_field, '')
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        if self.label_field in item:
            result['labels'] = torch.tensor(item[self.label_field], dtype=torch.long)
        
        return result


class ClassificationDataset(BaseTextDataset):
    """文本分类数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_field: str = 'text',
        label_field: str = 'label',
        num_labels: Optional[int] = None
    ):
        super().__init__(data, tokenizer, max_length, text_field, label_field)
        self.num_labels = num_labels
        
        if self.num_labels is None and self.label_field in data[0]:
            labels = set(item[self.label_field] for item in data if self.label_field in item)
            self.num_labels = len(labels)
            logger.info(f"检测到 {self.num_labels} 个类别")


class Seq2SeqDataset(Dataset):
    """序列到序列数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
        source_field: str = 'source',
        target_field: str = 'target'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_field = source_field
        self.target_field = target_field
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        source_text = item.get(self.source_field, '')
        target_text = item.get(self.target_field, '')
        
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }


class CausalLMDataset(Dataset):
    """因果语言模型数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_field: str = 'text',
        instruction_field: Optional[str] = None,
        response_field: Optional[str] = None
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.instruction_field = instruction_field
        self.response_field = response_field
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.instruction_field and self.response_field:
            instruction = item.get(self.instruction_field, '')
            response = item.get(self.response_field, '')
            text = f"{instruction}\n\n{response}"
        else:
            text = item.get(self.text_field, '')
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def safe_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    vocab_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    安全的collate函数，验证和修复token IDs
    
    Args:
        batch: 样本列表
        tokenizer: tokenizer对象（用于获取vocab_size）
        vocab_size: 词汇表大小（如果提供，优先使用）
        
    Returns:
        整理后的batch字典
    """
    import torch
    
    # 获取vocab_size
    if vocab_size is None and tokenizer is not None:
        # 尝试多种方式获取vocab_size
        if hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, '__len__'):
            vocab_size = len(tokenizer)
        elif hasattr(tokenizer, 'get_vocab'):
            vocab_size = len(tokenizer.get_vocab())
    
    # 默认的collate逻辑：将列表中的字典合并
    collated = {}
    if not batch:
        return collated
    
    # 获取所有键
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # 如果是tensor，stack它们
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError as e:
                # 如果shape不一致，尝试pad
                logger.warning(f"无法stack {key}，尝试padding: {e}")
                # 找到最大长度
                max_len = max(v.shape[-1] if v.dim() > 0 else 1 for v in values)
                # Pad所有tensor到相同长度
                padded_values = []
                for v in values:
                    if v.dim() == 1:
                        pad_len = max_len - v.shape[0]
                        if pad_len > 0:
                            pad_value = 0 if 'mask' not in key.lower() else 0
                            v = torch.cat([v, torch.full((pad_len,), pad_value, dtype=v.dtype)])
                        padded_values.append(v)
                    else:
                        padded_values.append(v)
                collated[key] = torch.stack(padded_values)
        elif isinstance(values[0], (list, tuple)):
            # 列表或元组，保持原样或转换为tensor
            collated[key] = values
        else:
            collated[key] = values
    
    # 验证和修复token IDs
    if vocab_size is not None:
        for key in ['input_ids', 'decoder_input_ids', 'labels']:
            if key in collated and isinstance(collated[key], torch.Tensor):
                tensor = collated[key]
                
                # 在CPU上检查，避免CUDA错误
                tensor_cpu = tensor.cpu()
                max_val = tensor_cpu.max().item()
                min_val = tensor_cpu.min().item()
                
                if key == 'labels':
                    # labels可以是-100（用于mask），只检查非-100的值
                    valid_tensor = tensor_cpu[tensor_cpu != -100]
                    if len(valid_tensor) > 0:
                        max_valid = valid_tensor.max().item()
                        min_valid = valid_tensor.min().item()
                        
                        if max_valid >= vocab_size or min_valid < 0:
                            # 写入调试日志文件，不刷屏
                            debug_logger = logging.getLogger('debug')
                            if debug_logger.handlers:
                                debug_logger.warning(
                                    f"⚠️  {key}包含非法值: [{min_valid}, {max_valid}] vs vocab_size={vocab_size}, "
                                    f"修复中..."
                                )
                            else:
                                logger.warning(
                                    f"⚠️  {key}包含非法值: [{min_valid}, {max_valid}] vs vocab_size={vocab_size}, "
                                    f"修复中..."
                                )
                            # 创建mask：非-100且超出范围的值
                            mask = (tensor_cpu != -100) & ((tensor_cpu < 0) | (tensor_cpu >= vocab_size))
                            tensor_cpu[mask] = -100
                            collated[key] = tensor_cpu.to(tensor.device) if tensor.is_cuda else tensor_cpu
                else:
                    # input_ids和decoder_input_ids必须在[0, vocab_size-1]范围内
                    if max_val >= vocab_size or min_val < 0:
                        # 写入调试日志文件，不刷屏
                        debug_logger = logging.getLogger('debug')
                        if debug_logger.handlers:
                            debug_logger.warning(
                                f"⚠️  {key}包含非法值: [{min_val}, {max_val}] vs vocab_size={vocab_size}, "
                                f"修复中..."
                            )
                        else:
                            logger.warning(
                                f"⚠️  {key}包含非法值: [{min_val}, {max_val}] vs vocab_size={vocab_size}, "
                                f"修复中..."
                            )
                        # Clamp到有效范围
                        tensor_cpu = torch.clamp(tensor_cpu, 0, vocab_size - 1)
                        collated[key] = tensor_cpu.to(tensor.device) if tensor.is_cuda else tensor_cpu
                        # 写入调试日志文件，不刷屏
                        debug_logger = logging.getLogger('debug')
                        if debug_logger.handlers:
                            debug_logger.info(f"   ✅ {key}修复后: min={tensor_cpu.min().item()}, max={tensor_cpu.max().item()}")
    
    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    vocab_size: Optional[int] = None,
    use_safe_collate: bool = True
) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        dataset: PyTorch Dataset
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存
        collate_fn: 自定义collate函数（如果提供，优先使用）
        tokenizer: tokenizer对象（用于safe_collate_fn获取vocab_size）
        vocab_size: 词汇表大小（用于safe_collate_fn）
        use_safe_collate: 是否使用安全的collate函数（默认True）
        
    Returns:
        DataLoader对象
    """
    # 如果没有提供collate_fn且use_safe_collate=True，使用safe_collate_fn
    if collate_fn is None and use_safe_collate:
        # 尝试从dataset获取tokenizer
        if tokenizer is None and hasattr(dataset, 'tokenizer'):
            tokenizer = dataset.tokenizer
        
        # 创建partial函数，绑定tokenizer和vocab_size
        from functools import partial
        collate_fn = partial(safe_collate_fn, tokenizer=tokenizer, vocab_size=vocab_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


# 示例用法
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoImageProcessor
    
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: VQA数据集
    vqa_data = [
        {
            "image": "path/to/image1.jpg",
            "question": "What color is the car?",
            "answer": "red"
        },
        {
            "image": "path/to/image2.jpg",
            "question": "How many people are in the image?",
            "answer": "three"
        }
    ]
    
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # vqa_dataset = VQADataset(
    #     data=vqa_data,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     image_root='data/images'
    # )
    
    # 示例2: Image Captioning数据集
    caption_data = [
        {
            "image": "path/to/image1.jpg",
            "caption": "A red car parked on the street"
        },
        {
            "image": "path/to/image2.jpg",
            "caption": ["Three people walking in the park", "Group of friends outdoors"]
        }
    ]
    
    # caption_dataset = ImageCaptioningDataset(
    #     data=caption_data,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     multiple_captions=True
    # )
    
    # dataloader = create_dataloader(vqa_dataset, batch_size=2)
    
    print("Dataset模块加载完成 - 支持VQA和Image Captioning")
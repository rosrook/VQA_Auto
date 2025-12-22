"""
模型加载器
支持多种模型类型（BLIP、CLIP等），保持向后兼容性
"""
import logging
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path
from transformers import (
    AutoModelForVisualQuestionAnswering,  # VQA任务（官方推荐）
    AutoModelForSeq2SeqLM,  # Seq2Seq任务（T5/BLIP生成任务）
    AutoModelForCausalLM,   # Causal LM任务（LLaMA/GPT）
    AutoModel,              # Encoder-only任务（检索任务等）
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """统一的模型加载器，支持多种模型类型"""
    
    # 支持的模型类型映射（使用官方推荐的Auto类）
    MODEL_REGISTRY = {
        'blip': {
            'vqa': AutoModelForVisualQuestionAnswering,  # 官方推荐：AutoModelForVisualQuestionAnswering
            'image_text_retrieval': AutoModel,  # 官方推荐：AutoModel
            'conditional_generation': AutoModelForSeq2SeqLM,  # 官方推荐：AutoModelForSeq2SeqLM
            'model_ids': [
                'Salesforce/blip-vqa-base',
                'Salesforce/blip-vqa-capfilt-large',
                'Salesforce/blip-image-captioning-base',
                'Salesforce/blip-image-captioning-large'
            ]
        },
        'auto': {
            'model_ids': []  # 支持所有模型
        }
    }
    
    def __init__(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        task: str = 'vqa',
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化模型加载器
        
        Args:
            model_name: HuggingFace模型ID或本地路径
            model_type: 模型类型 ('blip', 'auto')
            task: 任务类型 ('vqa', 'image_text_retrieval', 'conditional_generation')
            device: 设备 ('cuda', 'cpu', None=auto)
            **kwargs: 其他加载参数
                - torch_dtype: 数据类型 (torch.float32, torch.float16, torch.bfloat16)
                - device_map: 设备映射 ('auto', 'cuda:0', None)
                - load_in_8bit: 是否8bit量化
                - load_in_4bit: 是否4bit量化
                - trust_remote_code: 是否信任远程代码
                - low_cpu_mem_usage: 是否低CPU内存使用
        """
        self.model_name = model_name
        self.model_type = model_type or self._detect_model_type(model_name)
        self.task = task
        self.device = device or self._get_default_device()
        self.kwargs = kwargs
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        
        logger.info(f"初始化ModelLoader: {model_name} (类型: {self.model_type}, 任务: {task})")
    
    def _detect_model_type(self, model_name: str) -> str:
        """
        根据模型名称自动检测模型类型
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型类型
        """
        model_name_lower = model_name.lower()
        
        # 检查是否匹配已知的模型ID
        for model_type, model_info in self.MODEL_REGISTRY.items():
            if model_type == 'auto':
                continue
            for model_id in model_info.get('model_ids', []):
                if model_id.lower() in model_name_lower or model_name_lower in model_id.lower():
                    return model_type
        
        # 根据关键词检测
        if 'blip' in model_name_lower:
            return 'blip'
        else:
            return 'auto'  # 默认使用AutoModel
    
    def _get_default_device(self) -> str:
        """获取默认设备"""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def load(
        self,
        load_processor: bool = True,
        **override_kwargs
    ) -> Dict[str, Any]:
        """
        加载模型和processor
        
        Args:
            load_processor: 是否同时加载processor
            **override_kwargs: 覆盖初始化时的kwargs
            
        Returns:
            包含model, processor, tokenizer, image_processor的字典
        """
        # 合并kwargs
        load_kwargs = {**self.kwargs, **override_kwargs}
        
        logger.info(f"开始加载模型: {self.model_name}")
        
        # 加载模型
        self.model = self._load_model(load_kwargs)
        
        # 移动到指定设备
        if self.device and not load_kwargs.get('device_map'):
            self.model = self.model.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        result = {'model': self.model}
        
        # 加载processor（如果需要）
        if load_processor:
            processor_dict = self._load_processor()
            result.update(processor_dict)
            self.processor = processor_dict.get('processor')
            self.tokenizer = processor_dict.get('tokenizer')
            self.image_processor = processor_dict.get('image_processor')
        
        logger.info("模型加载完成")
        return result
    
    def _load_model(self, load_kwargs: Dict[str, Any]) -> PreTrainedModel:
        """
        加载模型
        
        Args:
            load_kwargs: 加载参数
            
        Returns:
            加载的模型
        """
        # 如果指定了类型，使用对应的模型类
        if self.model_type != 'auto':
            try:
                model_info = self.MODEL_REGISTRY.get(self.model_type)
                if model_info:
                    # 根据任务选择模型类
                    model_class = model_info.get(self.task)
                    if model_class:
                        logger.info(f"使用 {self.model_type} {self.task} 模型类: {model_class.__name__}")
                        return self._load_with_class(model_class, load_kwargs)
                    else:
                        logger.warning(f"{self.model_type} 不支持任务 {self.task}，使用AutoModel")
            except Exception as e:
                logger.warning(f"使用指定类型加载模型失败: {e}，回退到AutoModel")
        
        # 使用AutoModel（根据任务类型选择合适的类）
        # 默认使用 AutoModelForSeq2SeqLM（适用于VQA等Seq2Seq任务）
        # 注意：transformers >=4.5.0 已移除 AutoModelForConditionalGeneration
        if self.task in ['vqa', 'image_captioning', 'seq2seq']:
            logger.info("使用AutoModelForSeq2SeqLM（Seq2Seq任务）")
            return self._load_with_class(AutoModelForSeq2SeqLM, load_kwargs)
        elif self.task in ['causal_lm', 'text_generation']:
            logger.info("使用AutoModelForCausalLM（Causal LM任务）")
            return self._load_with_class(AutoModelForCausalLM, load_kwargs)
        else:
            logger.info("使用AutoModel（Encoder-only任务）")
            return self._load_with_class(AutoModel, load_kwargs)
    
    def _load_with_class(
        self,
        model_class: type,
        load_kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        """
        使用指定的模型类加载模型
        
        Args:
            model_class: 模型类
            load_kwargs: 加载参数
            
        Returns:
            加载的模型
        """
        # 准备加载参数
        safe_kwargs = {
            'trust_remote_code': load_kwargs.get('trust_remote_code', True),
            'low_cpu_mem_usage': load_kwargs.get('low_cpu_mem_usage', True),
        }
        
        # 数据类型
        # 注意：如果使用混合精度训练（AMP），模型参数应该是FP32
        # AMP会在forward时自动转换为FP16，但参数本身保持FP32
        # 如果模型参数是FP16/BFloat16，GradScaler无法工作
        if 'torch_dtype' in load_kwargs:
            safe_kwargs['torch_dtype'] = load_kwargs['torch_dtype']
            logger.info(f"使用指定的dtype: {load_kwargs['torch_dtype']}")
        elif self.device == 'cuda':
            # 默认使用float32，以便与AMP兼容
            # 如果用户需要FP16/BFloat16模型（不使用AMP），可以在load_kwargs中显式指定
            safe_kwargs['torch_dtype'] = torch.float32
            logger.info("默认使用float32加载模型（与AMP兼容）")
            # 如果需要FP16模型（不使用AMP），可以使用：
            # safe_kwargs['torch_dtype'] = torch.float16
            # 如果需要BFloat16模型（不使用AMP），可以使用：
            # safe_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 设备映射
        if self.device == 'cuda' and 'device_map' in load_kwargs:
            safe_kwargs['device_map'] = load_kwargs['device_map']
        elif self.device == 'cpu':
            safe_kwargs['device_map'] = None
        
        # 量化
        if load_kwargs.get('load_in_8bit', False):
            try:
                from transformers import BitsAndBytesConfig
                safe_kwargs['load_in_8bit'] = True
                logger.info("使用8bit量化")
            except ImportError:
                logger.warning("需要安装bitsandbytes来使用8bit量化: pip install bitsandbytes")
        
        if load_kwargs.get('load_in_4bit', False):
            try:
                from transformers import BitsAndBytesConfig
                safe_kwargs['load_in_4bit'] = True
                safe_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info("使用4bit量化")
            except ImportError:
                logger.warning("需要安装bitsandbytes来使用4bit量化: pip install bitsandbytes")
        
        # 加载模型
        # 注意：不要使用 torch.inference_mode() 或 torch.no_grad()
        # 这会导致模型无法跟踪梯度，训练时会报错：
        # RuntimeError: Inference tensors do not track version counter.
        try:
            model = model_class.from_pretrained(
                self.model_name,
                **safe_kwargs
            )
            logger.info(f"模型加载成功: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_processor(self) -> Dict[str, Any]:
        """
        加载processor，使用AutoProcessor（官方推荐方式）
        
        Returns:
            包含processor, tokenizer, image_processor的字典
        """
        logger.info("加载processor（使用AutoProcessor，官方推荐）...")
        
        try:
            # 使用AutoProcessor（官方推荐方式）
            # AutoProcessor会自动选择正确的processor类
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            result = {
                'processor': processor,
            }
            if hasattr(processor, 'tokenizer'):
                result['tokenizer'] = processor.tokenizer
            if hasattr(processor, 'image_processor'):
                result['image_processor'] = processor.image_processor
            
            # 记录processor类型信息
            processor_type = type(processor).__name__
            logger.info(f"Processor类型: {processor_type}")
            
            if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
                tokenizer_type = type(processor.tokenizer).__name__
                logger.info(f"Tokenizer类型: {tokenizer_type}")
            
            if hasattr(processor, 'image_processor') and processor.image_processor is not None:
                image_processor_type = type(processor.image_processor).__name__
                logger.info(f"ImageProcessor类型: {image_processor_type}")
            
            # 验证vocab_size匹配（如果模型已加载）
            if self.model is not None:
                self._verify_processor_model_match(processor, self.model)
            
            return result
        except Exception as e:
            logger.error(f"加载processor失败: {e}")
            logger.warning("返回空字典")
            return {}
    
    def _verify_processor_model_match(self, processor, model):
        """
        验证processor与模型的匹配性（vocab_size等）
        
        Args:
            processor: 加载的processor
            model: 加载的模型
        """
        try:
            # 检查tokenizer的vocab_size
            if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
                tokenizer_vocab_size = getattr(processor.tokenizer, 'vocab_size', None)
                
                # 检查模型的vocab_size
                model_vocab_size = None
                if hasattr(model, 'config'):
                    model_vocab_size = getattr(model.config, 'vocab_size', None)
                    # BLIP可能有text_config
                    if model_vocab_size is None and hasattr(model.config, 'text_config'):
                        text_config = model.config.text_config
                        model_vocab_size = getattr(text_config, 'vocab_size', None)
                
                if tokenizer_vocab_size is not None and model_vocab_size is not None:
                    if tokenizer_vocab_size != model_vocab_size:
                        logger.error("=" * 60)
                        logger.error("⚠️⚠️⚠️ 严重警告: vocab_size不匹配！")
                        logger.error(f"   Tokenizer vocab_size: {tokenizer_vocab_size}")
                        logger.error(f"   Model vocab_size: {model_vocab_size}")
                        logger.error("   这可能导致训练或推理错误！")
                        logger.error("=" * 60)
                    else:
                        logger.info(f"✅ vocab_size匹配: {tokenizer_vocab_size}")
                else:
                    logger.warning("无法验证vocab_size匹配（缺少vocab_size信息）")
        except Exception as e:
            logger.warning(f"验证processor-model匹配时出错: {e}")
    
    def get_model(self) -> PreTrainedModel:
        """获取模型"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load()")
        return self.model
    
    def get_processor(self):
        """获取processor"""
        if self.processor is None:
            raise RuntimeError("Processor未加载，请先调用load(load_processor=True)")
        return self.processor
    
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """获取tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer未加载，请先调用load(load_processor=True)")
        return self.tokenizer
    
    def get_image_processor(self):
        """获取image processor"""
        if self.image_processor is None:
            raise RuntimeError("Image processor未加载，请先调用load(load_processor=True)")
        return self.image_processor


def load_model(
    model_name: str,
    model_type: Optional[str] = None,
    task: str = 'vqa',
    device: Optional[str] = None,
    load_processor: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：加载模型
    
    Args:
        model_name: HuggingFace模型ID或本地路径
        model_type: 模型类型 ('blip', 'auto')
        task: 任务类型 ('vqa', 'image_text_retrieval', 'conditional_generation')
        device: 设备 ('cuda', 'cpu', None=auto)
        load_processor: 是否同时加载processor
        **kwargs: 其他加载参数
        
    Returns:
        包含model, processor等的字典
        
    Example:
        >>> result = load_model('Salesforce/blip-vqa-base', model_type='blip', task='vqa')
        >>> model = result['model']
        >>> processor = result['processor']
    """
    loader = ModelLoader(
        model_name=model_name,
        model_type=model_type,
        task=task,
        device=device,
        **kwargs
    )
    return loader.load(load_processor=load_processor)


# 示例用法
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: 加载BLIP VQA模型
    print("=" * 60)
    print("示例1: 加载BLIP VQA模型")
    print("=" * 60)
    
    try:
        result = load_model(
            model_name="Salesforce/blip-vqa-base",
            model_type="blip",
            task="vqa",
            device="cpu",  # 使用CPU避免GPU要求
            load_processor=True
        )
        
        model = result['model']
        processor = result.get('processor')
        tokenizer = result.get('tokenizer')
        image_processor = result.get('image_processor')
        
        print(f"\n✓ 模型加载成功: {type(model).__name__}")
        if processor:
            print(f"✓ Processor加载成功: {type(processor).__name__}")
        if tokenizer:
            print(f"✓ Tokenizer加载成功: {type(tokenizer).__name__}")
        if image_processor:
            print(f"✓ Image processor加载成功: {type(image_processor).__name__}")
            
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print("注意：这可能需要下载模型，请确保网络连接正常")
    
    print("\nModelLoader模块加载完成 - 支持BLIP等多种模型")


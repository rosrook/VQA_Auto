"""
模型工具函数
提供模型相关的实用工具函数，支持BLIP等多种模型
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForSeq2SeqLM,  # Seq2Seq任务（T5/BLIP/VQA）- transformers >=4.5.0推荐
    AutoModelForCausalLM,   # Causal LM任务（LLaMA/GPT）
    AutoModel,              # Encoder-only任务
    AutoProcessor,
)

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数
        
    Returns:
        参数字典，包含total, trainable等
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable = total
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def freeze_model(model: nn.Module, freeze_layers: Optional[List[str]] = None) -> nn.Module:
    """
    冻结模型的部分层
    
    Args:
        model: PyTorch模型
        freeze_layers: 要冻结的层名称列表（如果为None，冻结所有层）
        
    Returns:
        模型（原地修改）
    """
    if freeze_layers is None:
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        logger.info("冻结了所有层")
    else:
        # 冻结指定层
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
                logger.debug(f"冻结层: {name}")
        logger.info(f"冻结了 {len(freeze_layers)} 个层")
    
    return model


def unfreeze_model(model: nn.Module, unfreeze_layers: Optional[List[str]] = None) -> nn.Module:
    """
    解冻模型的部分层
    
    Args:
        model: PyTorch模型
        unfreeze_layers: 要解冻的层名称列表（如果为None，解冻所有层）
        
    Returns:
        模型（原地修改）
    """
    if unfreeze_layers is None:
        # 解冻所有层
        for param in model.parameters():
            param.requires_grad = True
        logger.info("解冻了所有层")
    else:
        # 解冻指定层
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
                logger.debug(f"解冻层: {name}")
        logger.info(f"解冻了 {len(unfreeze_layers)} 个层")
    
    return model


def get_device(model: nn.Module) -> torch.device:
    """
    获取模型所在的设备
    
    Args:
        model: PyTorch模型
        
    Returns:
        设备
    """
    return next(model.parameters()).device


def move_model_to_device(model: nn.Module, device: Union[str, torch.device]) -> nn.Module:
    """
    将模型移动到指定设备
    
    Args:
        model: PyTorch模型
        device: 目标设备
        
    Returns:
        模型（原地修改）
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    logger.info(f"模型已移动到设备: {device}")
    return model


def save_model(
    model: nn.Module,
    save_path: Union[str, Path],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    processor: Optional[Any] = None,
    **kwargs
) -> None:
    """
    保存模型
    
    Args:
        model: PyTorch模型
        save_path: 保存路径
        tokenizer: Tokenizer（可选）
        processor: Processor（可选）
        **kwargs: 其他保存参数
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    if isinstance(model, PreTrainedModel):
        model.save_pretrained(save_path, **kwargs)
        logger.info(f"模型已保存到: {save_path}")
    else:
        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
        logger.info(f"模型状态字典已保存到: {save_path / 'pytorch_model.bin'}")
    
    # 保存tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer已保存到: {save_path}")
    
    # 保存processor
    if processor is not None:
        if hasattr(processor, 'save_pretrained'):
            processor.save_pretrained(save_path)
            logger.info(f"Processor已保存到: {save_path}")
        else:
            logger.warning("Processor不支持save_pretrained方法")


def load_model_from_path(
    model_path: Union[str, Path],
    model_class: Optional[type] = None,
    device: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    从路径加载模型
    
    Args:
        model_path: 模型路径
        model_class: 模型类（如果为None，尝试自动检测）
        device: 设备
        **kwargs: 其他加载参数
        
    Returns:
        加载的模型
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    if model_class is None:
        # 尝试使用AutoModelForSeq2SeqLM（适用于VQA等Seq2Seq任务）
        # 注意：transformers >=4.5.0 已移除 AutoModelForConditionalGeneration
        from transformers import AutoModelForSeq2SeqLM
        model_class = AutoModelForSeq2SeqLM
    
    if issubclass(model_class, PreTrainedModel):
        model = model_class.from_pretrained(str(model_path), **kwargs)
    else:
        model = model_class()
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
        model.load_state_dict(state_dict)
    
    if device:
        model = model.to(device)
    
    model.eval()
    logger.info(f"模型已从 {model_path} 加载")
    return model


def generate_answer_blip(
    model: PreTrainedModel,  # 使用AutoModelForVisualQuestionAnswering（官方推荐）
    processor: AutoProcessor,  # 使用AutoProcessor（官方推荐）
    image: Image.Image,
    question: str,
    device: Optional[str] = None,
    max_length: int = 20,
    num_beams: int = 3,
    **kwargs
) -> str:
    """
    使用BLIP模型生成答案（VQA任务）
    
    Args:
        model: VQA模型（使用AutoModelForVisualQuestionAnswering，官方推荐）
        processor: Processor（使用AutoProcessor，官方推荐）
        image: PIL图像
        question: 问题文本
        device: 设备（如果为None，自动检测）
        max_length: 最大生成长度
        num_beams: beam search数量
        **kwargs: 其他生成参数
        
    Returns:
        生成的答案文本
    """
    if device is None:
        device = get_device(model)
    
    # 处理输入
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    # 生成答案
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
    
    # 解码答案
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer


def generate_caption_blip(
    model: PreTrainedModel,  # 使用AutoModelForSeq2SeqLM（官方推荐）
    processor: AutoProcessor,  # 使用AutoProcessor（官方推荐）
    image: Image.Image,
    device: Optional[str] = None,
    max_length: int = 20,
    num_beams: int = 3,
    **kwargs
) -> str:
    """
    使用BLIP模型生成图像描述
    
    Args:
        model: 生成模型（使用AutoModelForSeq2SeqLM，官方推荐）
        processor: Processor（使用AutoProcessor，官方推荐）
        image: PIL图像
        device: 设备（如果为None，自动检测）
        max_length: 最大生成长度
        num_beams: beam search数量
        **kwargs: 其他生成参数
        
    Returns:
        生成的描述文本
    """
    if device is None:
        device = get_device(model)
    
    # 处理输入
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 生成描述
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
    
    # 解码描述
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def prepare_inputs_for_blip(
    processor: AutoProcessor,  # 使用AutoProcessor（官方推荐）
    images: Union[Image.Image, List[Image.Image]],
    texts: Optional[Union[str, List[str]]] = None,
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    为BLIP模型准备输入
    
    Args:
        processor: Processor（使用AutoProcessor，官方推荐）
        images: 图像或图像列表
        texts: 文本或文本列表（可选）
        device: 设备
        
    Returns:
        处理后的输入字典
    """
    if texts is None:
        inputs = processor(images=images, return_tensors="pt")
    else:
        inputs = processor(images=images, text=texts, return_tensors="pt")
    
    if device:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    return inputs


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型信息字典
    """
    param_info = count_parameters(model)
    device = get_device(model)
    model_size_mb = get_model_size_mb(model)
    
    # 统计可训练层
    trainable_layers = []
    frozen_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
        else:
            frozen_layers.append(name)
    
    info = {
        'model_class': type(model).__name__,
        'total_parameters': param_info['total'],
        'trainable_parameters': param_info['trainable'],
        'non_trainable_parameters': param_info['non_trainable'],
        'total_parameters_millions': param_info['total_millions'],
        'trainable_parameters_millions': param_info['trainable_millions'],
        'model_size_mb': model_size_mb,
        'device': str(device),
        'num_trainable_layers': len(trainable_layers),
        'num_frozen_layers': len(frozen_layers),
        'is_training': model.training
    }
    
    return info


def print_model_summary(model: nn.Module, detailed: bool = False) -> None:
    """
    打印模型摘要
    
    Args:
        model: PyTorch模型
        detailed: 是否打印详细信息
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("模型摘要")
    print("=" * 60)
    print(f"模型类型: {info['model_class']}")
    print(f"总参数量: {info['total_parameters']:,} ({info['total_parameters_millions']:.2f}M)")
    print(f"可训练参数: {info['trainable_parameters']:,} ({info['trainable_parameters_millions']:.2f}M)")
    print(f"冻结参数: {info['non_trainable_parameters']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")
    print(f"设备: {info['device']}")
    print(f"训练模式: {info['is_training']}")
    print(f"可训练层数: {info['num_trainable_layers']}")
    print(f"冻结层数: {info['num_frozen_layers']}")
    
    if detailed:
        print("\n可训练层:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}: {param.shape}")
    
    print("=" * 60)


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    比较两个模型
    
    Args:
        model1: 第一个模型
        model2: 第二个模型
        
    Returns:
        比较结果字典
    """
    info1 = get_model_info(model1)
    info2 = get_model_info(model2)
    
    comparison = {
        'model1': info1,
        'model2': info2,
        'parameter_diff': info1['total_parameters'] - info2['total_parameters'],
        'size_diff_mb': info1['model_size_mb'] - info2['model_size_mb'],
        'same_class': info1['model_class'] == info2['model_class']
    }
    
    return comparison


# 示例用法
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 示例：创建简单的模型来演示工具函数
    print("=" * 60)
    print("模型工具函数示例")
    print("=" * 60)
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    
    # 统计参数
    param_info = count_parameters(model)
    print(f"\n参数量: {param_info['total']:,} ({param_info['total_millions']:.2f}M)")
    
    # 获取模型大小
    size_mb = get_model_size_mb(model)
    print(f"模型大小: {size_mb:.2f} MB")
    
    # 冻结部分层
    freeze_model(model, freeze_layers=['linear1'])
    print(f"\n冻结linear1后，可训练参数: {count_parameters(model, trainable_only=True)['trainable']:,}")
    
    # 打印模型摘要
    print_model_summary(model)
    
    print("\nModelUtils模块加载完成 - 提供模型相关的实用工具函数")


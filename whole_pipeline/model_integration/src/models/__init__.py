"""
模型模块
提供模型加载和工具函数
"""
from model_loader import ModelLoader, load_model
from model_utils import (
    count_parameters,
    get_model_size_mb,
    freeze_model,
    unfreeze_model,
    get_device,
    move_model_to_device,
    save_model,
    load_model_from_path,
    generate_answer_blip,
    generate_caption_blip,
    prepare_inputs_for_blip,
    get_model_info,
    print_model_summary,
    compare_models
)

__all__ = [
    # ModelLoader
    'ModelLoader',
    'load_model',
    # Model Utils
    'count_parameters',
    'get_model_size_mb',
    'freeze_model',
    'unfreeze_model',
    'get_device',
    'move_model_to_device',
    'save_model',
    'load_model_from_path',
    'generate_answer_blip',
    'generate_caption_blip',
    'prepare_inputs_for_blip',
    'get_model_info',
    'print_model_summary',
    'compare_models',
]


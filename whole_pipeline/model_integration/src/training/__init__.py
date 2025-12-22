"""
训练模块
提供模型训练、回调和评估功能
"""
from training.trainer import Trainer, create_trainer_from_config
from training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    TensorBoardCallback,
    ProgressBarCallback,
    CSVLoggerCallback
)
from training.evaluator import Evaluator, VQAEvaluator

__all__ = [
    # Trainer
    'Trainer',
    'create_trainer_from_config',
    # Callbacks
    'Callback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'LearningRateSchedulerCallback',
    'TensorBoardCallback',
    'ProgressBarCallback',
    'CSVLoggerCallback',
    # Evaluator
    'Evaluator',
    'VQAEvaluator',
]


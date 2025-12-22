"""
主管线
整合data、models、training、utils模块，提供统一的训练和推理接口
"""
import logging
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
import sys
import os

# 确保src目录在Python路径中（以便可以直接运行此脚本）
_script_dir = Path(__file__).parent.absolute()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# 导入各个模块
from data.data_pipeline import DataPipeline
from models.model_loader import load_model, ModelLoader
from models.model_utils import (
    print_model_summary, get_model_info, freeze_model,
    save_model as save_model_utils
)
from training.trainer import Trainer, create_trainer_from_config
from training.evaluator import Evaluator, VQAEvaluator
from training.callbacks import (
    EarlyStoppingCallback, ModelCheckpointCallback,
    LearningRateSchedulerCallback, TensorBoardCallback,
    ProgressBarCallback, CSVLoggerCallback
)
from utils.logger import Logger, TrainingLogger, setup_logger
from utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics, accuracy, exact_match, f1_score

logger = logging.getLogger(__name__)


class VQAPipeline:
    """VQA任务主管线"""
    
    def __init__(
        self,
        config_path: str,
        experiment_name: Optional[str] = None,
        **kwargs
    ):
        """
        初始化VQA管线
        
        Args:
            config_path: 配置文件路径
            experiment_name: 实验名称（用于日志和检查点）
            **kwargs: 其他参数
        """
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name or self._generate_experiment_name()
        
        # 初始化组件
        self.data_pipeline = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        self.trainer = None
        self.evaluator = None
        
        # 工具
        self.logger = None
        self.training_logger = None
        self.checkpoint_manager = None
        
        # 设置日志
        self._setup_logging(**kwargs)
        
        logger.info("=" * 60)
        logger.info(f"VQA Pipeline初始化: {self.experiment_name}")
        logger.info("=" * 60)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_type = self.config.get('task_type', 'vqa')
        return f"{task_type}_{timestamp}"
    
    def _setup_logging(self, **kwargs):
        """设置日志"""
        log_dir = kwargs.get('log_dir', 'logs')
        log_level = kwargs.get('log_level', self.config.get('logging', {}).get('level', 'INFO'))
        
        # 创建日志目录
        log_path = Path(log_dir) / self.experiment_name
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 设置全局日志
        self.logger = setup_logger(
            name=f"VQAPipeline_{self.experiment_name}",
            log_dir=str(log_path),
            level=log_level
        )
        
        # 创建训练日志记录器
        self.training_logger = TrainingLogger(
            log_dir=str(log_path),
            experiment_name=self.experiment_name
        )
        
        # 记录配置
        self.training_logger.log_config(self.config)
    
    def setup_data(self):
        """设置数据管线"""
        logger.info("设置数据管线...")
        self.data_pipeline = DataPipeline(str(self.config_path))
        self.data_pipeline.setup()
        logger.info("数据管线设置完成")
        return self.data_pipeline
    
    def setup_model(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        task: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """设置模型"""
        logger.info("设置模型...")
        
        # 从配置获取参数
        model_config = self.config.get('model', {})
        model_name = model_name or model_config.get('name', 'Salesforce/blip-vqa-base')
        model_type = model_type or model_config.get('type', 'blip')
        task = task or self.config.get('task_type', 'vqa')
        
        # 设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        model_result = load_model(
            model_name=model_name,
            model_type=model_type,
            task=task,
            device=device,
            load_processor=True,
            **kwargs
        )
        
        self.model = model_result['model']
        self.processor = model_result.get('processor')
        self.tokenizer = model_result.get('tokenizer')
        self.image_processor = model_result.get('image_processor')
        
        # 打印模型信息
        print_model_summary(self.model)
        
        logger.info("模型设置完成")
        return self.model
    
    def setup_training(
        self,
        **kwargs
    ):
        """设置训练"""
        logger.info("设置训练...")
        
        if self.data_pipeline is None:
            self.setup_data()
        
        if self.model is None:
            self.setup_model()
        
        # 获取数据加载器
        train_loader = self.data_pipeline.get_train_dataloader()
        val_loader = self.data_pipeline.get_val_dataloader() if 'validation' in self.data_pipeline.datasets else None
        
        # 训练配置
        training_config = self.config.get('training', {})
        
        # 优化器配置
        optimizer_config = training_config.get('optimizer', {})
        lr = optimizer_config.get('lr', 3e-5)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        optimizer_type = optimizer_config.get('type', 'adamw')
        
        from torch.optim import AdamW, Adam, SGD
        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler = None
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config:
            scheduler_type = scheduler_config.get('type', 'cosine')
            num_epochs = training_config.get('num_epochs', 3)
            
            from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
            if scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
            elif scheduler_type == 'step':
                scheduler = StepLR(
                    optimizer,
                    step_size=scheduler_config.get('step_size', 1),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 2)
                )
        
        # 创建评估器
        if self.processor:
            self.evaluator = VQAEvaluator(self.model, self.processor, device=self.model.device if hasattr(self.model, 'device') else 'cuda')
        else:
            self.evaluator = Evaluator(self.model, device=self.model.device if hasattr(self.model, 'device') else 'cuda')
        
        # 创建回调函数
        callbacks = []
        
        # 进度条
        callbacks.append(ProgressBarCallback(verbose=1))
        
        # 早停
        early_stopping_config = training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            callbacks.append(EarlyStoppingCallback(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 5),
                min_delta=early_stopping_config.get('min_delta', 0.0),
                mode=early_stopping_config.get('mode', 'min')
            ))
        
        # 检查点
        checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
        checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=training_config.get('max_checkpoints', 5),
            best_metric=early_stopping_config.get('monitor', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min')
        )
        
        callbacks.append(ModelCheckpointCallback(
            save_dir=str(checkpoint_dir),
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            save_best_only=training_config.get('save_best_only', True)
        ))
        
        # 学习率调度器回调
        if scheduler:
            callbacks.append(LearningRateSchedulerCallback(scheduler))
        
        # TensorBoard
        if training_config.get('use_tensorboard', False):
            callbacks.append(TensorBoardCallback(log_dir=str(checkpoint_dir / 'tensorboard')))
        
        # CSV日志
        callbacks.append(CSVLoggerCallback(filename=str(checkpoint_dir / 'training_log.csv')))
        
        # 冻结层
        freeze_config = training_config.get('freeze', {})
        if freeze_config.get('enabled', False):
            freeze_layers = freeze_config.get('layers', [])
            freeze_model(self.model, freeze_layers=freeze_layers if freeze_layers else None)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            evaluator=self.evaluator,
            num_epochs=training_config.get('num_epochs', 3),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            max_grad_norm=training_config.get('max_grad_norm'),
            fp16=training_config.get('fp16', False),
            save_dir=str(checkpoint_dir),
            **kwargs
        )
        
        logger.info("训练设置完成")
        return self.trainer
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            self.setup_training()
        
        logger.info("开始训练...")
        self.trainer.train()
        
        # 记录训练历史
        for epoch_logs in self.trainer.history:
            self.training_logger.log_epoch(
                epoch=epoch_logs['epoch'],
                train_metrics={k.replace('train_', ''): v for k, v in epoch_logs.items() if k.startswith('train_')},
                val_metrics={k.replace('val_', ''): v for k, v in epoch_logs.items() if k.startswith('val_')}
            )
        
        logger.info("训练完成")
    
    def evaluate(
        self,
        split: str = 'validation',
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            split: 数据集分割 ('train', 'validation', 'test')
            return_predictions: 是否返回预测结果
            
        Returns:
            评估结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用setup_model()")
        
        if self.evaluator is None:
            if self.processor:
                self.evaluator = VQAEvaluator(self.model, self.processor)
            else:
                self.evaluator = Evaluator(self.model)
        
        # 获取数据加载器
        if self.data_pipeline is None:
            self.setup_data()
        
        dataloader = self.data_pipeline.get_dataloader(split)
        
        # 评估
        logger.info(f"评估{split}集...")
        results = self.evaluator.evaluate(dataloader, return_predictions=return_predictions)
        
        # 记录结果
        self.training_logger.logger.log_metrics(results)
        
        return results
    
    def predict(
        self,
        image_path: str,
        question: str,
        **kwargs
    ) -> str:
        """
        预测单个样本
        
        Args:
            image_path: 图像路径
            question: 问题文本
            **kwargs: 其他参数
            
        Returns:
            预测答案
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型和processor未加载，请先调用setup_model()")
        
        from PIL import Image
        from models.model_utils import generate_answer_blip
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 生成答案
        answer = generate_answer_blip(
            model=self.model,
            processor=self.processor,
            image=image,
            question=question,
            **kwargs
        )
        
        return answer
    
    def save(self, save_dir: Optional[str] = None):
        """保存模型和配置"""
        if save_dir is None:
            save_dir = f"checkpoints/{self.experiment_name}"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if self.model:
            save_model_utils(
                model=self.model,
                save_path=str(save_path),
                tokenizer=self.tokenizer,
                processor=self.processor
            )
        
        # 保存配置
        config_file = save_path / 'config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"模型和配置已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用setup_model()")
        
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model
        )
        
        logger.info(f"检查点已加载: {checkpoint_path}")
        return checkpoint


def create_pipeline(
    config_path: str,
    experiment_name: Optional[str] = None,
    **kwargs
) -> VQAPipeline:
    """
    便捷函数：创建VQA管线
    
    Args:
        config_path: 配置文件路径
        experiment_name: 实验名称
        **kwargs: 其他参数
        
    Returns:
        VQAPipeline实例
        
    Example:
        >>> pipeline = create_pipeline('config/vqa_config.yaml')
        >>> pipeline.setup_data()
        >>> pipeline.setup_model()
        >>> pipeline.train()
    """
    return VQAPipeline(config_path=config_path, experiment_name=experiment_name, **kwargs)


# 示例用法
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("VQA Pipeline - 完整训练和推理管线")
    print("=" * 60)
    
    print("\n使用示例:")
    print("""
    # 1. 创建管线
    pipeline = create_pipeline('config/vqa_config.yaml', experiment_name='my_experiment')
    
    # 2. 设置数据
    pipeline.setup_data()
    
    # 3. 设置模型
    pipeline.setup_model()
    
    # 4. 设置训练
    pipeline.setup_training()
    
    # 5. 开始训练
    pipeline.train()
    
    # 6. 评估
    results = pipeline.evaluate(split='validation')
    
    # 7. 预测
    answer = pipeline.predict('path/to/image.jpg', 'What color is the car?')
    
    # 8. 保存
    pipeline.save()
    """)
    
    print("\nPipeline模块加载完成")


"""
训练器
集成data和model模块，提供完整的训练功能
"""
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
    LinearLR, ExponentialLR
)
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
from tqdm import tqdm

# 导入data和model模块（使用绝对导入）
from data.data_pipeline import DataPipeline
from models.model_loader import load_model
from models.model_utils import freeze_model, print_model_summary, get_model_info
from training.callbacks import (
    Callback, EarlyStoppingCallback, ModelCheckpointCallback,
    LearningRateSchedulerCallback, TensorBoardCallback,
    ProgressBarCallback, CSVLoggerCallback
)
from training.evaluator import Evaluator, VQAEvaluator

logger = logging.getLogger(__name__)


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        evaluator: Optional[Evaluator] = None,
        **kwargs
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器（可选）
            optimizer: 优化器（如果为None，会使用默认AdamW）
            scheduler: 学习率调度器（可选）
            device: 设备
            callbacks: 回调函数列表
            evaluator: 评估器（如果为None，会创建默认的）
            **kwargs: 其他参数
                - num_epochs: 训练轮数
                - gradient_accumulation_steps: 梯度累积步数
                - max_grad_norm: 梯度裁剪
                - fp16: 是否使用混合精度训练
                - save_dir: 保存目录
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设备
        self.device = device or next(model.parameters()).device
        self.model = self.model.to(self.device)
        
        # 优化器
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=3e-5)
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # 回调函数
        self.callbacks = callbacks or []
        
        # 评估器
        self.evaluator = evaluator or Evaluator(model, device=self.device)
        
        # 训练参数
        self.num_epochs = kwargs.get('num_epochs', 3)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = kwargs.get('max_grad_norm', None)
        self.fp16 = kwargs.get('fp16', False)
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.history = []
        
        # 混合精度训练
        if self.fp16:
            try:
                from torch.cuda.amp import autocast, GradScaler
                self.scaler = GradScaler()
                self.use_amp = True
                logger.info("启用混合精度训练（FP16）")
            except ImportError:
                logger.warning("无法启用混合精度训练，需要CUDA支持")
                self.use_amp = False
        else:
            self.use_amp = False
        
        # 创建保存目录
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.info(f"验证样本数: {len(val_dataloader.dataset)}")
    
    def train(self):
        """开始训练"""
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        
        # 打印模型信息
        print_model_summary(self.model)
        
        # 调用训练开始回调
        self._call_callbacks('on_train_begin')
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                # 调用epoch开始回调
                self._call_callbacks('on_epoch_begin', epoch=epoch)
                
                # 训练一个epoch
                train_logs = self._train_epoch()
                
                # 验证（如果有验证集）
                val_logs = {}
                if self.val_dataloader:
                    val_logs = self._validate()
                
                # 合并日志
                epoch_logs = {**train_logs, **val_logs}
                epoch_logs['epoch'] = epoch
                
                # 记录历史
                self.history.append(epoch_logs)
                
                # 调用epoch结束回调
                self._call_callbacks('on_epoch_end', epoch=epoch, logs=epoch_logs)
                
                # 检查是否早停
                if self.should_stop:
                    logger.info("训练提前停止")
                    break
        
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        
        finally:
            # 调用训练结束回调
            self._call_callbacks('on_train_end')
            logger.info("训练完成")
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 调用batch开始回调
            self._call_callbacks('on_batch_begin', batch=batch_idx)
            
            try:
                # 准备输入（包含验证和修复）
                batch = self._prepare_batch(batch)
                
                # 前向传播
                loss = self._train_step(batch)
            except RuntimeError as e:
                error_str = str(e)
                if "CUDA" in error_str or "device-side assert" in error_str or "index" in error_str.lower():
                    # CUDA错误，可能是索引越界，打印详细信息
                    logger.error(f"CUDA错误在batch {batch_idx}: {e}")
                    logger.error("Batch内容:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.error(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                            if 'ids' in key.lower() or 'mask' in key.lower():
                                logger.error(f"    min={value.min().item()}, max={value.max().item()}")
                                if value.numel() < 100:  # 如果tensor不大，打印所有值
                                    logger.error(f"    values={value.tolist()}")
                        else:
                            logger.error(f"  {key}: {type(value)}")
                    raise
                else:
                    raise
            
            # 累积损失
            total_loss += loss
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # 调用batch结束回调
            batch_logs = {'loss': loss}
            self._call_callbacks('on_batch_end', batch=batch_idx, logs=batch_logs)
            
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'train_loss': avg_loss}
    
    def _train_step(self, batch: Dict[str, Any]) -> float:
        """执行一个训练步骤"""
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                loss = loss / self.gradient_accumulation_steps
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            loss = loss / self.gradient_accumulation_steps
        
        # 反向传播
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm is not None:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            else:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _validate(self) -> Dict[str, float]:
        """验证"""
        logger.info("开始验证...")
        
        # 调用验证开始回调
        self._call_callbacks('on_validation_begin')
        
        # 评估
        val_logs = self.evaluator.evaluate(self.val_dataloader)
        
        # 添加val_前缀
        val_logs = {f'val_{k}': v for k, v in val_logs.items()}
        
        # 调用验证结束回调
        self._call_callbacks('on_validation_end', logs=val_logs)
        
        return val_logs
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备batch，移动到设备并验证tensor shapes和值
        
        注意：BLIP等模型对attention_mask的形状和值有严格要求
        """
        prepared_batch = {}
        
        # 首先验证关键字段
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"input_ids应该是torch.Tensor，得到{type(input_ids)}")
            
            # 验证input_ids shape
            if input_ids.dim() != 2:
                raise ValueError(f"input_ids应该是2D tensor [batch_size, seq_len]，得到shape {input_ids.shape}")
            
            batch_size, seq_len = input_ids.shape
            
            # 验证attention_mask（如果存在）
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
                if not isinstance(attention_mask, torch.Tensor):
                    raise TypeError(f"attention_mask应该是torch.Tensor，得到{type(attention_mask)}")
                
                # 验证attention_mask shape
                if attention_mask.shape != input_ids.shape:
                    logger.warning(
                        f"attention_mask shape {attention_mask.shape} 与 input_ids shape {input_ids.shape} 不匹配，"
                        f"尝试修复..."
                    )
                    # 尝试修复：如果维度不匹配，尝试reshape或重新创建
                    if attention_mask.dim() == 1 and len(attention_mask) == seq_len:
                        # 如果是1D且长度匹配，扩展到batch维度
                        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
                    elif attention_mask.dim() == 2 and attention_mask.size(0) == batch_size:
                        # 如果batch维度匹配但seq_len不匹配，重新创建
                        if attention_mask.size(1) != seq_len:
                            # 重新创建attention_mask：非padding位置为1
                            pad_id = getattr(self.model.config, 'pad_token_id', None) if hasattr(self.model, 'config') else None
                            if pad_id is None:
                                # 如果没有pad_token_id，假设所有非0位置都是有效token
                                attention_mask = (input_ids != 0).long()
                            else:
                                attention_mask = (input_ids != pad_id).long()
                    else:
                        # 完全重新创建
                        pad_id = getattr(self.model.config, 'pad_token_id', None) if hasattr(self.model, 'config') else None
                        if pad_id is None:
                            attention_mask = (input_ids != 0).long()
                        else:
                            attention_mask = (input_ids != pad_id).long()
                    
                    logger.info(f"修复后的attention_mask shape: {attention_mask.shape}")
                
                # 验证attention_mask值（应该是0或1）
                unique_values = torch.unique(attention_mask)
                invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
                if len(invalid_values) > 0:
                    logger.warning(
                        f"attention_mask包含非法值: {invalid_values.tolist()}，"
                        f"将clamp到[0, 1]范围"
                    )
                    attention_mask = torch.clamp(attention_mask, 0, 1).long()
                
                prepared_batch['attention_mask'] = attention_mask.to(self.device)
            
            # 验证labels（如果存在）
            if 'labels' in batch:
                labels = batch['labels']
                if isinstance(labels, torch.Tensor):
                    if labels.shape != input_ids.shape:
                        logger.warning(
                            f"labels shape {labels.shape} 与 input_ids shape {input_ids.shape} 不匹配"
                        )
                        # 尝试修复：如果维度不匹配
                        if labels.dim() == 1 and len(labels) == seq_len:
                            labels = labels.unsqueeze(0).expand(batch_size, -1)
                        elif labels.dim() == 2 and labels.size(0) == batch_size and labels.size(1) != seq_len:
                            # 如果seq_len不匹配，可能需要padding或truncation
                            logger.error(f"无法修复labels shape不匹配: {labels.shape} vs {input_ids.shape}")
                            raise ValueError(f"labels shape不匹配: {labels.shape} vs {input_ids.shape}")
                    prepared_batch['labels'] = labels.to(self.device)
            
            prepared_batch['input_ids'] = input_ids.to(self.device)
        
        # 处理其他字段
        for key, value in batch.items():
            if key not in prepared_batch:  # 避免重复处理
                if isinstance(value, torch.Tensor):
                    prepared_batch[key] = value.to(self.device)
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    # 处理tensor列表（如pixel_values的batch）
                    prepared_batch[key] = [v.to(self.device) for v in value]
                else:
                    prepared_batch[key] = value
        
        # 最终验证：确保所有tensor都在同一设备上
        for key, value in prepared_batch.items():
            if isinstance(value, torch.Tensor) and value.device != self.device:
                logger.warning(f"{key}不在正确设备上: {value.device} vs {self.device}，移动到{self.device}")
                prepared_batch[key] = value.to(self.device)
        
        return prepared_batch
    
    def _call_callbacks(self, method_name: str, **kwargs):
        """调用回调函数"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(self, **kwargs)
                except Exception as e:
                    logger.error(f"回调函数 {callback.__class__.__name__}.{method_name} 执行失败: {e}")
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            **kwargs
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"检查点已加载: {filepath}")


def create_trainer_from_config(
    data_config_path: str,
    model_name: str,
    model_type: Optional[str] = None,
    task: str = 'vqa',
    **kwargs
) -> Trainer:
    """
    从配置文件创建训练器
    
    Args:
        data_config_path: 数据配置文件路径
        model_name: 模型名称
        model_type: 模型类型
        task: 任务类型
        **kwargs: 其他训练参数
        
    Returns:
        Trainer实例
    """
    # 1. 加载数据
    logger.info("加载数据...")
    pipeline = DataPipeline(data_config_path)
    pipeline.setup()
    train_loader = pipeline.get_train_dataloader()
    val_loader = pipeline.get_val_dataloader() if 'validation' in pipeline.datasets else None
    
    # 2. 加载模型
    logger.info("加载模型...")
    model_result = load_model(
        model_name=model_name,
        model_type=model_type,
        task=task,
        device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        load_processor=True
    )
    model = model_result['model']
    processor = model_result.get('processor')
    
    # 3. 配置优化器
    optimizer_config = kwargs.get('optimizer', {})
    lr = optimizer_config.get('lr', 3e-5)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    optimizer_type = optimizer_config.get('type', 'adamw')
    
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 4. 配置学习率调度器
    scheduler = None
    scheduler_config = kwargs.get('scheduler', {})
    if scheduler_config:
        scheduler_type = scheduler_config.get('type', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('num_epochs', 3)
            )
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
    
    # 5. 创建评估器
    evaluator = None
    if task == 'vqa' and processor:
        evaluator = VQAEvaluator(model, processor, device=kwargs.get('device'))
    else:
        evaluator = Evaluator(model, device=kwargs.get('device'))
    
    # 6. 创建回调函数
    callbacks = []
    
    # 进度条
    callbacks.append(ProgressBarCallback(verbose=1))
    
    # 早停
    if kwargs.get('early_stopping', {}).get('enabled', False):
        callbacks.append(EarlyStoppingCallback(
            monitor=kwargs['early_stopping'].get('monitor', 'val_loss'),
            patience=kwargs['early_stopping'].get('patience', 5)
        ))
    
    # 模型检查点
    save_dir = kwargs.get('save_dir', 'checkpoints')
    callbacks.append(ModelCheckpointCallback(
        save_dir=save_dir,
        monitor=kwargs.get('checkpoint_monitor', 'val_loss'),
        save_best_only=kwargs.get('save_best_only', True)
    ))
    
    # 学习率调度器回调
    if scheduler:
        callbacks.append(LearningRateSchedulerCallback(scheduler))
    
    # TensorBoard（如果启用）
    if kwargs.get('use_tensorboard', False):
        callbacks.append(TensorBoardCallback(log_dir=f'{save_dir}/tensorboard'))
    
    # CSV日志
    callbacks.append(CSVLoggerCallback(filename=f'{save_dir}/training_log.csv'))
    
    # 7. 冻结层（如果配置）
    freeze_config = kwargs.get('freeze', {})
    if freeze_config.get('enabled', False):
        freeze_layers = freeze_config.get('layers', [])
        freeze_model(model, freeze_layers=freeze_layers if freeze_layers else None)
    
    # 8. 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=kwargs.get('device'),
        callbacks=callbacks,
        evaluator=evaluator,
        num_epochs=kwargs.get('num_epochs', 3),
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
        max_grad_norm=kwargs.get('max_grad_norm'),
        fp16=kwargs.get('fp16', False),
        save_dir=save_dir
    )
    
    return trainer


# 示例用法
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Trainer模块加载完成 - 提供完整的训练功能")
    print("\n使用示例:")
    print("""
    from training.trainer import create_trainer_from_config
    
    trainer = create_trainer_from_config(
        data_config_path='config/vqa_config.yaml',
        model_name='Salesforce/blip-vqa-base',
        model_type='blip',
        task='vqa',
        num_epochs=3,
        fp16=True
    )
    
    trainer.train()
    """)


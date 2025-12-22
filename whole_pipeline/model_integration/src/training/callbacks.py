"""
训练回调函数
提供训练过程中的各种回调功能
"""
import logging
import torch
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import json
import time

logger = logging.getLogger(__name__)


class Callback(ABC):
    """回调基类"""
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs):
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时调用"""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: int, **kwargs):
        """每个batch开始时调用"""
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float], **kwargs):
        """每个batch结束时调用"""
        pass
    
    def on_validation_begin(self, trainer: Any, **kwargs):
        """验证开始时调用"""
        pass
    
    def on_validation_end(self, trainer: Any, logs: Dict[str, float], **kwargs):
        """验证结束时调用"""
        pass


class EarlyStoppingCallback(Callback):
    """早停回调"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        初始化早停回调
        
        Args:
            monitor: 监控的指标名称
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            mode: 'min' 或 'max'
            restore_best_weights: 是否恢复最佳权重
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_epoch = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = None
        
    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时初始化"""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0
        self.best_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时检查"""
        if self.monitor not in logs:
            logger.warning(f"监控指标 {self.monitor} 不在logs中")
            return
        
        current_score = logs[self.monitor]
        
        # 判断是否改善
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict().copy()
                logger.info(f"Epoch {epoch}: {self.monitor} 改善到 {current_score:.4f}")
        else:
            self.wait += 1
            logger.info(f"Epoch {epoch}: {self.monitor} = {current_score:.4f}, "
                       f"最佳 = {self.best_score:.4f}, 等待 {self.wait}/{self.patience}")
        
        # 检查是否早停
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            logger.info(f"早停触发！Epoch {epoch}, 最佳epoch: {self.best_epoch}, "
                       f"最佳{self.monitor}: {self.best_score:.4f}")
            
            # 恢复最佳权重
            if self.restore_best_weights and self.best_weights is not None:
                trainer.model.load_state_dict(self.best_weights)
                logger.info("已恢复最佳权重")
    
    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束时恢复最佳权重"""
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            logger.info(f"训练结束，已恢复最佳权重（Epoch {self.best_epoch}）")


class ModelCheckpointCallback(Callback):
    """模型检查点回调"""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 1,
        save_top_k: int = 3,
        filename: str = 'checkpoint-{epoch:03d}-{val_loss:.4f}.pt'
    ):
        """
        初始化检查点回调
        
        Args:
            save_dir: 保存目录
            monitor: 监控的指标
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最佳模型
            save_frequency: 保存频率（每N个epoch）
            save_top_k: 保存top-k个最佳模型
            filename: 文件名模板
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.save_top_k = save_top_k
        self.filename = filename
        
        self.best_scores = []  # 保存top-k的最佳分数
        self.checkpoint_paths = []  # 保存的检查点路径
        
    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时初始化"""
        self.best_scores = []
        self.checkpoint_paths = []
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时保存检查点"""
        # 检查保存频率
        if epoch % self.save_frequency != 0:
            return
        
        # 获取监控指标
        if self.monitor not in logs:
            logger.warning(f"监控指标 {self.monitor} 不在logs中，跳过保存")
            return
        
        current_score = logs[self.monitor]
        
        # 判断是否保存
        should_save = False
        if self.save_best_only:
            # 只保存最佳模型
            if len(self.best_scores) == 0:
                should_save = True
            elif self.mode == 'min':
                should_save = current_score < max(self.best_scores)
            else:
                should_save = current_score > min(self.best_scores)
        else:
            # 每个epoch都保存
            should_save = True
        
        if should_save:
            # 生成文件名
            filename = self.filename.format(epoch=epoch, **logs)
            checkpoint_path = self.save_dir / filename
            
            # 保存模型
            self._save_checkpoint(trainer, checkpoint_path, epoch, logs)
            
            # 更新top-k列表
            if self.save_best_only:
                self.best_scores.append(current_score)
                self.checkpoint_paths.append(checkpoint_path)
                
                # 只保留top-k
                if len(self.best_scores) > self.save_top_k:
                    # 找到最差的
                    if self.mode == 'min':
                        worst_idx = self.best_scores.index(max(self.best_scores))
                    else:
                        worst_idx = self.best_scores.index(min(self.best_scores))
                    
                    # 删除最差的检查点
                    worst_path = self.checkpoint_paths.pop(worst_idx)
                    if worst_path.exists():
                        worst_path.unlink()
                        logger.info(f"删除旧的检查点: {worst_path}")
                    
                    self.best_scores.pop(worst_idx)
            
            logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _save_checkpoint(
        self,
        trainer: Any,
        checkpoint_path: Path,
        epoch: int,
        logs: Dict[str, float]
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
            'logs': logs,
            'model_config': getattr(trainer, 'model_config', {}),
        }
        
        # 如果有scheduler，也保存
        if hasattr(trainer, 'scheduler') and trainer.scheduler:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)


class LearningRateSchedulerCallback(Callback):
    """学习率调度回调"""
    
    def __init__(self, scheduler: Any):
        """
        初始化学习率调度回调
        
        Args:
            scheduler: 学习率调度器
        """
        self.scheduler = scheduler
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float], **kwargs):
        """每个batch结束时更新学习率（用于step-based scheduler）"""
        if hasattr(self.scheduler, 'step'):
            if hasattr(self.scheduler, 'step_on_batch') and self.scheduler.step_on_batch:
                self.scheduler.step()
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时更新学习率（用于epoch-based scheduler）"""
        if hasattr(self.scheduler, 'step'):
            if not (hasattr(self.scheduler, 'step_on_batch') and self.scheduler.step_on_batch):
                self.scheduler.step()
        
        # 记录当前学习率
        if trainer.optimizer:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            logs['lr'] = current_lr
            logger.debug(f"Epoch {epoch} 学习率: {current_lr:.2e}")


class TensorBoardCallback(Callback):
    """TensorBoard回调"""
    
    def __init__(self, log_dir: str):
        """
        初始化TensorBoard回调
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard日志目录: {self.log_dir}")
        except ImportError:
            logger.warning("未安装tensorboard，TensorBoardCallback将不会工作")
            logger.warning("安装命令: pip install tensorboard")
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时初始化"""
        if self.writer:
            # 记录模型图（如果可能）
            try:
                # 尝试记录模型结构
                pass  # 需要实际的输入示例
            except:
                pass
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float], **kwargs):
        """每个batch结束时记录"""
        if self.writer:
            global_step = getattr(trainer, 'global_step', batch)
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, global_step)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时记录"""
        if self.writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch/{key}', value, epoch)
    
    def on_validation_end(self, trainer: Any, logs: Dict[str, float], **kwargs):
        """验证结束时记录"""
        if self.writer:
            epoch = getattr(trainer, 'current_epoch', 0)
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'val/{key}', value, epoch)
    
    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束时关闭writer"""
        if self.writer:
            self.writer.close()


class ProgressBarCallback(Callback):
    """进度条回调"""
    
    def __init__(self, verbose: int = 1):
        """
        初始化进度条回调
        
        Args:
            verbose: 详细程度 (0=静默, 1=进度条, 2=详细信息)
        """
        self.verbose = verbose
        self.epoch_start_time = None
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs):
        """每个epoch开始时"""
        if self.verbose >= 1:
            self.epoch_start_time = time.time()
            print(f"\nEpoch {epoch}/{getattr(trainer, 'num_epochs', '?')}")
            print("-" * 60)
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float], **kwargs):
        """每个batch结束时"""
        if self.verbose >= 2:
            log_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, float)])
            print(f"  Batch {batch}: {log_str}")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时"""
        if self.verbose >= 1:
            elapsed_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            log_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, float)])
            print(f"Epoch {epoch} 完成 - {log_str} (耗时: {elapsed_time:.2f}s)")
    
    def on_validation_end(self, trainer: Any, logs: Dict[str, float], **kwargs):
        """验证结束时"""
        if self.verbose >= 1:
            log_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, float)])
            print(f"验证结果 - {log_str}")


class CSVLoggerCallback(Callback):
    """CSV日志回调"""
    
    def __init__(self, filename: str):
        """
        初始化CSV日志回调
        
        Args:
            filename: CSV文件名
        """
        self.filename = Path(filename)
        self.file = None
        self.fieldnames = None
        self.writer = None
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时初始化CSV文件"""
        import csv
        self.file = open(self.filename, 'w', newline='')
        self.fieldnames = ['epoch', 'train_loss', 'val_loss']
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], **kwargs):
        """每个epoch结束时写入CSV"""
        row = {'epoch': epoch}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        self.writer.writerow(row)
        self.file.flush()
    
    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束时关闭文件"""
        if self.file:
            self.file.close()


# 示例用法
if __name__ == "__main__":
    print("Callbacks模块加载完成 - 提供训练回调功能")


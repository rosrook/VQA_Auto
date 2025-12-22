"""
检查点管理
提供模型检查点的保存、加载和管理功能
"""
import logging
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        best_metric: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最多保存的检查点数量
            best_metric: 用于判断最佳模型的指标
            mode: 'min' 或 'max'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.mode = mode
        
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            epoch: epoch编号
            metrics: 指标字典
            metadata: 元数据
            is_best: 是否是最佳模型
            filename: 文件名（如果为None，自动生成）
            
        Returns:
            保存的检查点路径
        """
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_best:
                filename = f"best_checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
            else:
                filename = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 记录检查点信息
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best,
            'timestamp': checkpoint['timestamp']
        }
        self.checkpoints.append(checkpoint_info)
        
        # 更新最佳检查点
        if is_best or self._is_better(metrics):
            if self.best_checkpoint and self.best_checkpoint.exists():
                # 删除旧的最佳检查点
                old_best = self.checkpoint_dir / f"best_{self.best_checkpoint.name}"
                if old_best.exists():
                    old_best.unlink()
            
            # 创建最佳检查点的副本
            best_path = self.checkpoint_dir / f"best_{filename}"
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint = best_path
            self.best_score = metrics.get(self.best_metric, 0.0) if metrics else 0.0
            logger.info(f"更新最佳检查点: {best_path}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        # 保存检查点索引
        self._save_checkpoint_index()
        
        return checkpoint_path
    
    def _is_better(self, metrics: Optional[Dict[str, float]]) -> bool:
        """判断当前检查点是否更好"""
        if metrics is None or self.best_metric not in metrics:
            return False
        
        current_score = metrics[self.best_metric]
        if self.mode == 'min':
            return current_score < self.best_score
        else:
            return current_score > self.best_score
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # 按epoch排序，保留最新的
        self.checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
        
        # 删除多余的检查点（保留最佳检查点）
        to_remove = self.checkpoints[self.max_checkpoints:]
        for checkpoint_info in to_remove:
            if not checkpoint_info['is_best']:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"删除旧检查点: {checkpoint_path}")
        
        # 更新列表
        self.checkpoints = self.checkpoints[:self.max_checkpoints]
    
    def _save_checkpoint_index(self):
        """保存检查点索引"""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        index_data = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': str(self.best_checkpoint) if self.best_checkpoint else None,
            'best_score': self.best_score
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径（如果为None，加载最新的）
            load_best: 是否加载最佳检查点
            map_location: 设备映射
            
        Returns:
            检查点字典
        """
        if load_best and self.best_checkpoint:
            checkpoint_path = str(self.best_checkpoint)
        elif checkpoint_path is None:
            # 加载最新的检查点
            if self.checkpoints:
                checkpoint_path = self.checkpoints[-1]['path']
            else:
                raise ValueError("没有可用的检查点")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        return checkpoint
    
    def load_model(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        strict: bool = True
    ) -> torch.nn.Module:
        """
        加载模型权重
        
        Args:
            model: 模型实例
            checkpoint_path: 检查点路径
            load_best: 是否加载最佳检查点
            strict: 是否严格匹配
            
        Returns:
            加载权重后的模型
        """
        checkpoint = self.load(checkpoint_path=checkpoint_path, load_best=load_best)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"模型权重已加载，epoch: {checkpoint.get('epoch', 'unknown')}")
        return model
    
    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> torch.optim.Optimizer:
        """加载优化器状态"""
        checkpoint = self.load(checkpoint_path=checkpoint_path, load_best=load_best)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("优化器状态已加载")
        return optimizer
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点路径"""
        if self.checkpoints:
            return Path(self.checkpoints[-1]['path'])
        return None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """获取最佳检查点路径"""
        return self.best_checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        return self.checkpoints.copy()


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False,
    **kwargs
) -> Path:
    """
    便捷函数：保存检查点
    
    Args:
        checkpoint_dir: 检查点目录
        model: 模型
        epoch: epoch编号
        optimizer: 优化器
        scheduler: 学习率调度器
        metrics: 指标
        is_best: 是否最佳
        **kwargs: 其他参数
        
    Returns:
        保存的检查点路径
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=metrics,
        is_best=is_best,
        **kwargs
    )


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型（如果提供，会加载权重）
        optimizer: 优化器（如果提供，会加载状态）
        scheduler: 学习率调度器（如果提供，会加载状态）
        map_location: 设备映射
        
    Returns:
        检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("模型权重已加载")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("优化器状态已加载")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("学习率调度器状态已加载")
    
    return checkpoint


# 示例用法
if __name__ == "__main__":
    import torch.nn as nn
    
    # 创建示例模型
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 创建检查点管理器
    manager = CheckpointManager(
        checkpoint_dir="checkpoints",
        max_checkpoints=3
    )
    
    # 保存检查点
    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={'loss': 0.5, 'accuracy': 0.8},
        is_best=True
    )
    
    print(f"检查点已保存: {checkpoint_path}")
    print("Checkpoint模块加载完成")


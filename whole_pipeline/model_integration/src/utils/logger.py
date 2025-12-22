"""
日志工具
提供统一的日志记录功能，支持文件和控制台输出
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class Logger:
    """统一的日志记录器"""
    
    def __init__(
        self,
        name: str = "VQA_Pipeline",
        log_dir: Optional[str] = None,
        level: str = "INFO",
        format_string: Optional[str] = None,
        file_logging: bool = True,
        console_logging: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件目录（如果为None，不保存文件）
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            format_string: 自定义格式字符串
            file_logging: 是否保存到文件
            console_logging: 是否输出到控制台
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.level = getattr(logging, level.upper(), logging.INFO)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()  # 清除已有处理器
        
        # 设置格式
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # 控制台处理器
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if file_logging and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建日志文件（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    def debug(self, message: str, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化消息"""
        if kwargs:
            extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} ({extra_info})"
        return message
    
    def log_dict(self, data: Dict[str, Any], level: str = "INFO"):
        """记录字典数据"""
        message = json.dumps(data, indent=2, ensure_ascii=False)
        getattr(self.logger, level.lower())(message)
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """记录指标"""
        if epoch is not None:
            self.info(f"Epoch {epoch} Metrics:")
        else:
            self.info("Metrics:")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")


class TrainingLogger:
    """训练专用日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None
    ):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        
        # 创建主日志记录器
        self.logger = Logger(
            name=f"Training_{self.experiment_name}",
            log_dir=str(self.log_dir),
            level="INFO"
        )
        
        # 训练历史
        self.history = []
        self.history_file = self.log_dir / f"{self.experiment_name}_history.json"
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """记录一个epoch的指标"""
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics,
            'val': val_metrics or {}
        }
        
        self.history.append(epoch_data)
        
        # 记录到日志
        self.logger.info(f"Epoch {epoch} completed")
        self.logger.log_metrics(train_metrics, epoch=epoch)
        if val_metrics:
            self.logger.log_metrics(val_metrics, epoch=epoch)
        
        # 保存历史
        self.save_history()
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Configuration saved")
        self.logger.log_dict(config)
    
    def save_history(self):
        """保存训练历史"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def load_history(self) -> List[Dict[str, Any]]:
        """加载训练历史"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        return self.history
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> Optional[int]:
        """获取最佳epoch"""
        if not self.history:
            return None
        
        best_value = float('inf') if mode == 'min' else float('-inf')
        best_epoch = None
        
        for epoch_data in self.history:
            val_metrics = epoch_data.get('val', {})
            if metric in val_metrics:
                value = val_metrics[metric]
                if (mode == 'min' and value < best_value) or (mode == 'max' and value > best_value):
                    best_value = value
                    best_epoch = epoch_data['epoch']
        
        return best_epoch


# 全局日志记录器实例
_global_logger: Optional[Logger] = None


def setup_logger(
    name: str = "VQA_Pipeline",
    log_dir: Optional[str] = None,
    level: str = "INFO"
) -> Logger:
    """
    设置全局日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        
    Returns:
        Logger实例
    """
    global _global_logger
    _global_logger = Logger(name=name, log_dir=log_dir, level=level)
    return _global_logger


def get_logger() -> Logger:
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


# 示例用法
if __name__ == "__main__":
    # 创建日志记录器
    logger = Logger(
        name="TestLogger",
        log_dir="logs",
        level="INFO"
    )
    
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    # 训练日志记录器
    training_logger = TrainingLogger(
        log_dir="logs/training",
        experiment_name="test_experiment"
    )
    
    training_logger.log_epoch(
        epoch=1,
        train_metrics={'loss': 0.5, 'accuracy': 0.8},
        val_metrics={'loss': 0.4, 'accuracy': 0.85}
    )
    
    print("Logger模块加载完成")


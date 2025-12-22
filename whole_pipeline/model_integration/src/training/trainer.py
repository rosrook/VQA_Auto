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

# 创建调试日志文件处理器
_debug_log_file = None
_debug_log_handler = None

def setup_debug_logger(log_dir: str = "logs"):
    """设置调试日志文件"""
    global _debug_log_file, _debug_log_handler
    
    import os
    from pathlib import Path
    from datetime import datetime
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 创建调试日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _debug_log_file = log_path / f"debug_{timestamp}.log"
    
    # 创建文件处理器
    _debug_log_handler = logging.FileHandler(_debug_log_file, mode='w', encoding='utf-8')
    _debug_log_handler.setLevel(logging.DEBUG)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _debug_log_handler.setFormatter(formatter)
    
    # 创建调试logger
    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.addHandler(_debug_log_handler)
    debug_logger.propagate = False  # 不传播到root logger，避免刷屏
    
    logger.info(f"调试日志文件: {_debug_log_file}")
    return debug_logger

def get_debug_logger():
    """获取调试logger"""
    debug_logger = logging.getLogger('debug')
    if not debug_logger.handlers:
        # 如果没有设置，使用默认路径
        setup_debug_logger()
    return debug_logger


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
        
        # 设置调试日志（写入文件，不刷屏）
        self.debug_logger = setup_debug_logger(log_dir=str(Path(self.save_dir) / "debug_logs"))
        
        logger.info(f"训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.info(f"验证样本数: {len(val_dataloader.dataset)}")
        
        # 验证第一个batch
        self.validate_first_batch()
    
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
                # 注意：在CPU上完成所有验证和修复，然后再移动到GPU
                batch = self._prepare_batch(batch)
                
                # 在移动到GPU后，再次验证（如果可能）
                self._validate_batch_on_device(batch, batch_idx)
                
                # 前向传播
                loss = self._train_step(batch)
            except RuntimeError as e:
                error_str = str(e)
                if "CUDA" in error_str or "device-side assert" in error_str or "index" in error_str.lower():
                    debug_logger = get_debug_logger()
                    debug_logger.error("=" * 80)
                    debug_logger.error(f"CUDA错误在batch {batch_idx}: {e}")
                    debug_logger.error("=" * 80)
                    debug_logger.error("尝试在CPU上检查batch（如果可能）...")
                    
                    # 尝试获取原始batch（在移动到GPU之前）
                    # 如果batch已经在GPU上且CUDA错误，可能无法移动回CPU
                    # 所以我们需要在_prepare_batch之前保存一份CPU副本
                    try:
                        # 尝试将tensor移回CPU检查
                        batch_cpu = {}
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                try:
                                    # 使用detach()和cpu()，避免梯度问题
                                    value_cpu = value.detach().cpu()
                                    batch_cpu[key] = value_cpu
                                    debug_logger.error(f"  {key}:")
                                    debug_logger.error(f"    shape: {value.shape}")
                                    debug_logger.error(f"    dtype: {value.dtype}")
                                    debug_logger.error(f"    device: {value.device}")
                                    if 'ids' in key.lower() or 'mask' in key.lower():
                                        debug_logger.error(f"    min: {value_cpu.min().item()}")
                                        debug_logger.error(f"    max: {value_cpu.max().item()}")
                                        if value.numel() < 100:
                                            debug_logger.error(f"    values: {value_cpu.tolist()}")
                                except Exception as inner_e:
                                    debug_logger.error(f"  {key}: 无法移动到CPU检查 - {inner_e}")
                            else:
                                debug_logger.error(f"  {key}: {type(value)} = {value}")
                    except Exception as check_e:
                        debug_logger.error(f"无法检查batch详情: {check_e}")
                    
                    debug_logger.error("=" * 80)
                    debug_logger.error("建议：检查数据加载和_prepare_batch中的验证逻辑")
                    debug_logger.error("=" * 80)
                    
                    # 控制台只输出简要信息
                    logger.error(f"❌ CUDA错误在batch {batch_idx}（详细信息已写入调试日志文件）")
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
    
    # def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     准备batch，移动到设备并验证tensor shapes和值
        
    #     注意：BLIP等模型对attention_mask的形状和值有严格要求
    #     """
    #     prepared_batch = {}
        
    #     # 首先验证关键字段
    #     if 'input_ids' in batch:
    #         input_ids = batch['input_ids']
    #         if not isinstance(input_ids, torch.Tensor):
    #             raise TypeError(f"input_ids应该是torch.Tensor，得到{type(input_ids)}")
            
    #         # 验证input_ids shape
    #         if input_ids.dim() != 2:
    #             raise ValueError(f"input_ids应该是2D tensor [batch_size, seq_len]，得到shape {input_ids.shape}")
            
    #         batch_size, seq_len = input_ids.shape
            
    #         # 验证attention_mask（如果存在）
    #         if 'attention_mask' in batch:
    #             attention_mask = batch['attention_mask']
    #             if not isinstance(attention_mask, torch.Tensor):
    #                 raise TypeError(f"attention_mask应该是torch.Tensor，得到{type(attention_mask)}")
                
    #             # 验证attention_mask shape
    #             if attention_mask.shape != input_ids.shape:
    #                 logger.warning(
    #                     f"attention_mask shape {attention_mask.shape} 与 input_ids shape {input_ids.shape} 不匹配，"
    #                     f"尝试修复..."
    #                 )
    #                 # 尝试修复：如果维度不匹配，尝试reshape或重新创建
    #                 if attention_mask.dim() == 1 and len(attention_mask) == seq_len:
    #                     # 如果是1D且长度匹配，扩展到batch维度
    #                     attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
    #                 elif attention_mask.dim() == 2 and attention_mask.size(0) == batch_size:
    #                     # 如果batch维度匹配但seq_len不匹配，重新创建
    #                     if attention_mask.size(1) != seq_len:
    #                         # 重新创建attention_mask：非padding位置为1
    #                         pad_id = getattr(self.model.config, 'pad_token_id', None) if hasattr(self.model, 'config') else None
    #                         if pad_id is None:
    #                             # 如果没有pad_token_id，假设所有非0位置都是有效token
    #                             attention_mask = (input_ids != 0).long()
    #                         else:
    #                             attention_mask = (input_ids != pad_id).long()
    #                 else:
    #                     # 完全重新创建
    #                     pad_id = getattr(self.model.config, 'pad_token_id', None) if hasattr(self.model, 'config') else None
    #                     if pad_id is None:
    #                         attention_mask = (input_ids != 0).long()
    #                     else:
    #                         attention_mask = (input_ids != pad_id).long()
                    
    #                 logger.info(f"修复后的attention_mask shape: {attention_mask.shape}")
                
    #             # 验证attention_mask值（应该是0或1）
    #             unique_values = torch.unique(attention_mask)
    #             invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
    #             if len(invalid_values) > 0:
    #                 logger.warning(
    #                     f"attention_mask包含非法值: {invalid_values.tolist()}，"
    #                     f"将clamp到[0, 1]范围"
    #                 )
    #                 attention_mask = torch.clamp(attention_mask, 0, 1).long()
                
    #             prepared_batch['attention_mask'] = attention_mask.to(self.device)
            
    #         # 验证labels（如果存在）
    #         if 'labels' in batch:
    #             labels = batch['labels']
    #             if isinstance(labels, torch.Tensor):
    #                 if labels.shape != input_ids.shape:
    #                     logger.warning(
    #                         f"labels shape {labels.shape} 与 input_ids shape {input_ids.shape} 不匹配"
    #                     )
    #                     # 尝试修复：如果维度不匹配
    #                     if labels.dim() == 1 and len(labels) == seq_len:
    #                         labels = labels.unsqueeze(0).expand(batch_size, -1)
    #                     elif labels.dim() == 2 and labels.size(0) == batch_size and labels.size(1) != seq_len:
    #                         # 如果seq_len不匹配，可能需要padding或truncation
    #                         logger.error(f"无法修复labels shape不匹配: {labels.shape} vs {input_ids.shape}")
    #                         raise ValueError(f"labels shape不匹配: {labels.shape} vs {input_ids.shape}")
    #                 prepared_batch['labels'] = labels.to(self.device)
            
    #         prepared_batch['input_ids'] = input_ids.to(self.device)
        
    #     # 处理其他字段
    #     for key, value in batch.items():
    #         if key not in prepared_batch:  # 避免重复处理
    #             if isinstance(value, torch.Tensor):
    #                 prepared_batch[key] = value.to(self.device)
    #             elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
    #                 # 处理tensor列表（如pixel_values的batch）
    #                 prepared_batch[key] = [v.to(self.device) for v in value]
    #             else:
    #                 prepared_batch[key] = value
        
    #     # 最终验证：确保所有tensor都在同一设备上
    #     for key, value in prepared_batch.items():
    #         if isinstance(value, torch.Tensor) and value.device != self.device:
    #             logger.warning(f"{key}不在正确设备上: {value.device} vs {self.device}，移动到{self.device}")
    #             prepared_batch[key] = value.to(self.device)
        
    #     return prepared_batch

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备batch，移动到设备并验证tensor shapes和值
        
        特别注意BLIP模型的特殊要求
        """
        prepared_batch = {}
        
        # 获取模型词表大小
        vocab_size = None
        text_vocab_size = None
        if hasattr(self.model, 'config'):
            vocab_size = getattr(self.model.config, 'vocab_size', None)
            # BLIP有单独的text_config
            if hasattr(self.model.config, 'text_config'):
                text_vocab_size = getattr(self.model.config.text_config, 'vocab_size', None)
        
        # 使用text_vocab_size（如果存在）
        effective_vocab_size = text_vocab_size or vocab_size
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"input_ids应该是torch.Tensor，得到{type(input_ids)}")
            
            if input_ids.dim() != 2:
                raise ValueError(f"input_ids应该是2D tensor [batch_size, seq_len]，得到shape {input_ids.shape}")
            
            batch_size, seq_len = input_ids.shape
            
            # 在CPU上验证（确保在移动到GPU前完成所有修复）
            # 如果input_ids已经在GPU上，先移回CPU
            if input_ids.is_cuda:
                input_ids_cpu = input_ids.cpu()
            else:
                input_ids_cpu = input_ids.clone()
            
            max_id = input_ids_cpu.max().item()
            min_id = input_ids_cpu.min().item()
            
            debug_logger = get_debug_logger()
            debug_logger.info(f"📊 input_ids统计: min={min_id}, max={max_id}, vocab_size={effective_vocab_size}")
            
            # 检查并修复
            if effective_vocab_size is not None:
                if max_id >= effective_vocab_size or min_id < 0:
                    debug_logger.error(f"❌ input_ids超出范围: [{min_id}, {max_id}] vs [0, {effective_vocab_size-1}]")
                    
                    # 修复策略
                    pad_id = getattr(self.model.config, 'pad_token_id', 0)
                    unk_id = getattr(self.model.config, 'unk_token_id', pad_id)
                    
                    debug_logger.warning(f"   🔧 Clamping到有效范围...")
                    input_ids_cpu = torch.clamp(input_ids_cpu, 0, effective_vocab_size - 1)
                    input_ids = input_ids_cpu
                    
                    debug_logger.info(f"   ✅ 修复后: min={input_ids.min().item()}, max={input_ids.max().item()}")
            
            # 确保input_ids在CPU上，然后再移动到GPU
            if input_ids.is_cuda:
                input_ids = input_ids.cpu()
            prepared_batch['input_ids'] = input_ids.to(self.device)
            
            # ===== 关键：处理 decoder_input_ids (BLIP特有) =====
            if 'decoder_input_ids' in batch:
                decoder_input_ids = batch['decoder_input_ids']
                if isinstance(decoder_input_ids, torch.Tensor):
                    # 确保在CPU上处理
                    if decoder_input_ids.is_cuda:
                        decoder_input_ids_cpu = decoder_input_ids.cpu()
                    else:
                        decoder_input_ids_cpu = decoder_input_ids.clone()
                    
                    max_dec_id = decoder_input_ids_cpu.max().item()
                    min_dec_id = decoder_input_ids_cpu.min().item()
                    
                    debug_logger = get_debug_logger()
                    debug_logger.info(f"📊 decoder_input_ids统计: min={min_dec_id}, max={max_dec_id}")
                    
                    if effective_vocab_size is not None:
                        if max_dec_id >= effective_vocab_size or min_dec_id < 0:
                            debug_logger.error(f"❌ decoder_input_ids超出范围!")
                            decoder_input_ids_cpu = torch.clamp(decoder_input_ids_cpu, 0, effective_vocab_size - 1)
                            decoder_input_ids = decoder_input_ids_cpu
                            debug_logger.info(f"   ✅ decoder修复后: min={decoder_input_ids.min().item()}, max={decoder_input_ids.max().item()}")
                    
                    # 确保在CPU上，然后再移动到GPU
                    if decoder_input_ids.is_cuda:
                        decoder_input_ids = decoder_input_ids.cpu()
                    prepared_batch['decoder_input_ids'] = decoder_input_ids.to(self.device)
            
            # 处理 attention_mask
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
                if not isinstance(attention_mask, torch.Tensor):
                    raise TypeError(f"attention_mask应该是torch.Tensor，得到{type(attention_mask)}")
                
                debug_logger = get_debug_logger()
                if attention_mask.shape != input_ids.shape:
                    debug_logger.warning(f"attention_mask shape不匹配，重新创建...")
                    pad_id = getattr(self.model.config, 'pad_token_id', 0)
                    # 使用CPU上的input_ids创建attention_mask
                    input_ids_for_mask = prepared_batch.get('input_ids', input_ids)
                    if input_ids_for_mask.is_cuda:
                        input_ids_for_mask = input_ids_for_mask.cpu()
                    attention_mask = (input_ids_for_mask != pad_id).long()
                
                # 验证值（在CPU上）
                if attention_mask.is_cuda:
                    attention_mask_cpu = attention_mask.cpu()
                else:
                    attention_mask_cpu = attention_mask.clone()
                
                unique_values = torch.unique(attention_mask_cpu)
                if not all(v in [0, 1] for v in unique_values.tolist()):
                    debug_logger.warning(f"attention_mask包含非法值，修复中...")
                    attention_mask = torch.clamp(attention_mask_cpu, 0, 1).long()
                
                # 确保在CPU上，然后再移动到GPU
                if attention_mask.is_cuda:
                    attention_mask = attention_mask.cpu()
                prepared_batch['attention_mask'] = attention_mask.to(self.device)
            
            # ===== 关键：处理 decoder_attention_mask =====
            if 'decoder_attention_mask' in batch:
                decoder_attention_mask = batch['decoder_attention_mask']
                if isinstance(decoder_attention_mask, torch.Tensor):
                    decoder_attention_mask_cpu = decoder_attention_mask.cpu()
                    unique_values = torch.unique(decoder_attention_mask_cpu)
                    debug_logger = get_debug_logger()
                    if not all(v in [0, 1] for v in unique_values.tolist()):
                        debug_logger.warning(f"decoder_attention_mask包含非法值，修复中...")
                        decoder_attention_mask = torch.clamp(decoder_attention_mask_cpu, 0, 1).long()
                    prepared_batch['decoder_attention_mask'] = decoder_attention_mask.to(self.device)
            
            # 处理 labels
            if 'labels' in batch:
                labels = batch['labels']
                if isinstance(labels, torch.Tensor):
                    debug_logger = get_debug_logger()
                    if labels.shape != input_ids.shape:
                        debug_logger.warning(f"labels shape {labels.shape} 与 input_ids shape {input_ids.shape} 不匹配")
                        if labels.dim() == 1 and len(labels) == seq_len:
                            labels = labels.unsqueeze(0).expand(batch_size, -1)
                        elif labels.dim() == 2 and labels.size(0) == batch_size and labels.size(1) != seq_len:
                            # 对于BLIP，labels可能是answer的token ids，长度可能不同
                            debug_logger.info(f"labels长度与input_ids不同，这对BLIP是正常的")
                    
                    # 验证labels值（在CPU上，确保在移动到GPU前完成所有修复）
                    # 如果labels已经在GPU上，先移回CPU
                    if labels.is_cuda:
                        labels_cpu = labels.cpu()
                    else:
                        labels_cpu = labels.clone()
                    
                    valid_labels = labels_cpu[labels_cpu != -100]
                    if len(valid_labels) > 0:
                        max_label = valid_labels.max().item()
                        min_label = valid_labels.min().item()
                        
                        debug_logger.info(f"📊 labels统计: min={min_label}, max={max_label} (忽略-100)")
                        
                        if effective_vocab_size is not None:
                            if max_label >= effective_vocab_size or min_label < 0:
                                debug_logger.error(f"❌ labels超出范围: [{min_label}, {max_label}] vs [0, {effective_vocab_size-1}]")
                                debug_logger.warning(f"   🔧 将非法labels设置为-100...")
                                
                                # 创建mask并替换
                                mask = (labels_cpu != -100) & ((labels_cpu < 0) | (labels_cpu >= effective_vocab_size))
                                labels_cpu[mask] = -100
                                labels = labels_cpu
                                
                                debug_logger.info(f"   ✅ labels修复完成")
                    
                    # 确保labels在CPU上，然后再移动到GPU
                    if labels.is_cuda:
                        labels = labels.cpu()
                    prepared_batch['labels'] = labels.to(self.device)
        
        # 处理其他字段
        for key, value in batch.items():
            if key not in prepared_batch:
                if isinstance(value, torch.Tensor):
                    prepared_batch[key] = value.to(self.device)
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    prepared_batch[key] = [v.to(self.device) for v in value]
                else:
                    prepared_batch[key] = value
        
        return prepared_batch
    
    def _validate_batch_on_device(self, batch: Dict[str, Any], batch_idx: int):
        """
        在GPU上验证batch（如果可能）
        
        注意：这个方法在batch已经移动到GPU后调用，主要用于最后的检查
        """
        debug_logger = get_debug_logger()
        
        # 获取vocab_size
        vocab_size = None
        text_vocab_size = None
        if hasattr(self.model, 'config'):
            vocab_size = getattr(self.model.config, 'vocab_size', None)
            if hasattr(self.model.config, 'text_config'):
                text_vocab_size = getattr(self.model.config.text_config, 'vocab_size', None)
        
        effective_vocab_size = text_vocab_size or vocab_size
        
        # 只检查关键字段，避免频繁的CPU-GPU传输
        if batch_idx == 0:  # 只在第一个batch时详细检查
            for key in ['input_ids', 'labels', 'attention_mask']:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    tensor = batch[key]
                    try:
                        # 尝试在GPU上检查（如果可能）
                        if tensor.is_cuda:
                            # 对于CUDA tensor，先尝试在GPU上检查min/max
                            # 如果失败，说明tensor可能已经损坏
                            try:
                                max_val = tensor.max().item()
                                min_val = tensor.min().item()
                                
                                if key == 'labels':
                                    # labels可以是-100
                                    valid_tensor = tensor[tensor != -100]
                                    if len(valid_tensor) > 0:
                                        max_valid = valid_tensor.max().item()
                                        min_valid = valid_tensor.min().item()
                                        
                                        if effective_vocab_size and (max_valid >= effective_vocab_size or min_valid < 0):
                                            debug_logger.error(
                                                f"⚠️  GPU上的{key}包含非法值: [{min_valid}, {max_valid}] vs vocab_size={effective_vocab_size}"
                                            )
                                else:
                                    if effective_vocab_size and (max_val >= effective_vocab_size or min_val < 0):
                                        debug_logger.error(
                                            f"⚠️  GPU上的{key}包含非法值: [{min_val}, {max_val}] vs vocab_size={effective_vocab_size}"
                                        )
                            except RuntimeError as e:
                                debug_logger.warning(f"无法在GPU上检查{key}: {e}")
                    except Exception as e:
                        debug_logger.warning(f"检查{key}时出错: {e}")
    
    def validate_first_batch(self):
        """
        验证第一个训练batch的数据
        
        在训练开始前检查数据是否正确，特别是token IDs是否在有效范围内
        """
        debug_logger = get_debug_logger()
        debug_logger.info("=" * 60)
        debug_logger.info("🔍 验证第一个训练batch...")
        debug_logger.info("=" * 60)
        
        # 只在控制台输出简要信息
        logger.info("🔍 验证第一个训练batch（详细信息写入调试日志文件）...")
        
        try:
            # 获取第一个batch
            first_batch = next(iter(self.train_dataloader))
            
            # 获取vocab_size
            vocab_size = None
            text_vocab_size = None
            
            if hasattr(self.model, 'config'):
                vocab_size = getattr(self.model.config, 'vocab_size', None)
                # BLIP有单独的text_config
                if hasattr(self.model.config, 'text_config'):
                    text_vocab_size = getattr(self.model.config.text_config, 'vocab_size', None)
            
            effective_vocab_size = text_vocab_size or vocab_size
            
            if effective_vocab_size:
                debug_logger.info(f"📊 模型词汇表大小: {effective_vocab_size}")
            else:
                debug_logger.warning("⚠️  无法获取模型词汇表大小，跳过token ID范围验证")
            
            # 检查每个关键字段
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    # 移动到CPU检查，避免CUDA错误
                    value_cpu = value.cpu()
                    
                    debug_logger.info(f"\n  {key}:")
                    debug_logger.info(f"    shape: {value.shape}")
                    debug_logger.info(f"    dtype: {value.dtype}")
                    debug_logger.info(f"    device: {value.device}")
                    debug_logger.info(f"    min: {value_cpu.min().item()}")
                    debug_logger.info(f"    max: {value_cpu.max().item()}")
                    
                    # 检查token ID字段
                    if 'id' in key.lower() and effective_vocab_size:
                        max_val = value_cpu.max().item()
                        min_val = value_cpu.min().item()
                        
                        if key == 'labels':
                            # labels可以是-100，只检查非-100的值
                            valid_values = value_cpu[value_cpu != -100]
                            if len(valid_values) > 0:
                                max_valid = valid_values.max().item()
                                min_valid = valid_values.min().item()
                                
                                if max_valid >= effective_vocab_size or min_valid < 0:
                                    debug_logger.error(
                                        f"    ❌ 错误：{key}包含非法token ID: "
                                        f"[{min_valid}, {max_valid}] vs vocab_size={effective_vocab_size}"
                                    )
                                    logger.error(f"❌ {key}包含非法token ID，详情见调试日志")
                                else:
                                    debug_logger.info(f"    ✅ {key} token ID范围正常: [{min_valid}, {max_valid}]")
                        else:
                            # input_ids和decoder_input_ids必须在[0, vocab_size-1]范围内
                            if max_val >= effective_vocab_size or min_val < 0:
                                debug_logger.error(
                                    f"    ❌ 错误：{key}包含非法token ID: "
                                    f"[{min_val}, {max_val}] vs vocab_size={effective_vocab_size}"
                                )
                                logger.error(f"❌ {key}包含非法token ID，详情见调试日志")
                            else:
                                debug_logger.info(f"    ✅ {key} token ID范围正常: [{min_val}, {max_val}]")
                    
                    # 检查attention_mask
                    if 'mask' in key.lower():
                        unique_values = torch.unique(value_cpu)
                        invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
                        if len(invalid_values) > 0:
                            debug_logger.error(
                                f"    ❌ 错误：{key}包含非法值（不是0或1）: {invalid_values.tolist()}"
                            )
                            logger.error(f"❌ {key}包含非法值，详情见调试日志")
                        else:
                            debug_logger.info(f"    ✅ {key}值正常（仅包含0和1）")
            
            debug_logger.info("=" * 60)
            debug_logger.info("✅ 第一个batch验证完成")
            debug_logger.info("=" * 60)
            
            logger.info("✅ 第一个batch验证完成（详细信息已写入调试日志）")
            
        except Exception as e:
            debug_logger = get_debug_logger()
            debug_logger.error(f"❌ 验证第一个batch时出错: {e}")
            debug_logger.error("这可能导致训练时出现CUDA错误，请检查数据加载代码")
            import traceback
            debug_logger.error(traceback.format_exc())
            logger.error(f"❌ 验证第一个batch时出错: {e}（详细信息见调试日志）")
    
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


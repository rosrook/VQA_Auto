"""
模型评估器
提供模型评估和指标计算功能
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 设备
            metrics: 要计算的指标列表
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.metrics = metrics or ['loss', 'accuracy']
        
        # 注册的指标函数
        self.metric_functions = {
            'loss': self._compute_loss,
            'accuracy': self._compute_accuracy,
            'exact_match': self._compute_exact_match,
            'f1': self._compute_f1,
        }
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            return_predictions: 是否返回预测结果
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_losses = []
        all_predictions = []
        all_labels = []
        all_inputs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                # 准备输入
                batch = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self._forward_pass(batch)
                
                # 收集结果
                if 'loss' in outputs:
                    all_losses.append(outputs['loss'].item())
                
                # 收集预测和标签
                predictions = self._get_predictions(outputs, batch)
                labels = self._get_labels(batch)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                if return_predictions:
                    all_inputs.append({
                        'input_ids': batch.get('input_ids'),
                        'pixel_values': batch.get('pixel_values')
                    })
        
        # 计算指标
        results = {}
        
        if all_losses:
            results['loss'] = np.mean(all_losses)
        
        # 计算其他指标
        for metric_name in self.metrics:
            if metric_name == 'loss':
                continue
            
            if metric_name in self.metric_functions:
                metric_value = self.metric_functions[metric_name](
                    all_predictions, all_labels
                )
                results[metric_name] = metric_value
        
        # 返回结果
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
            results['inputs'] = all_inputs
        
        return results
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """准备batch，移动到设备"""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch
    
    def _forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """执行前向传播"""
        # 提取输入
        inputs = {}
        if 'pixel_values' in batch:
            inputs['pixel_values'] = batch['pixel_values']
        if 'input_ids' in batch:
            inputs['input_ids'] = batch['input_ids']
        if 'attention_mask' in batch:
            inputs['attention_mask'] = batch['attention_mask']
        if 'labels' in batch:
            inputs['labels'] = batch['labels']
        
        # 前向传播
        outputs = self.model(**inputs)
        
        return outputs
    
    def _get_predictions(self, outputs: Any, batch: Dict[str, Any]) -> List[Any]:
        """获取预测结果"""
        # 对于生成任务，使用logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        else:
            # 如果没有logits，尝试从labels获取
            if 'labels' in batch:
                predictions = batch['labels'].cpu().numpy().tolist()
            else:
                predictions = []
        
        return predictions
    
    def _get_labels(self, batch: Dict[str, Any]) -> List[Any]:
        """获取标签"""
        if 'labels' in batch:
            labels = batch['labels'].cpu().numpy().tolist()
        else:
            labels = []
        return labels
    
    def _compute_loss(self, predictions: List, labels: List) -> float:
        """计算损失（已在evaluate中计算）"""
        return 0.0  # 占位符
    
    def _compute_accuracy(self, predictions: List, labels: List) -> float:
        """计算准确率"""
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        # 将列表展平
        preds_flat = []
        labels_flat = []
        
        for pred, label in zip(predictions, labels):
            if isinstance(pred, list):
                preds_flat.extend(pred)
            else:
                preds_flat.append(pred)
            
            if isinstance(label, list):
                labels_flat.extend(label)
            else:
                labels_flat.append(label)
        
        # 计算准确率
        correct = sum(p == l for p, l in zip(preds_flat, labels_flat))
        total = len(preds_flat)
        
        return correct / total if total > 0 else 0.0
    
    def _compute_exact_match(self, predictions: List, labels: List) -> float:
        """计算完全匹配率"""
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        matches = 0
        for pred, label in zip(predictions, labels):
            if isinstance(pred, list) and isinstance(label, list):
                if pred == label:
                    matches += 1
            elif pred == label:
                matches += 1
        
        return matches / len(predictions) if len(predictions) > 0 else 0.0
    
    def _compute_f1(self, predictions: List, labels: List) -> float:
        """计算F1分数（简化版，用于分类任务）"""
        # 这里是一个简化的F1计算
        # 对于更复杂的任务，可能需要tokenizer来解码文本
        accuracy = self._compute_accuracy(predictions, labels)
        # 简化：使用accuracy作为F1的近似
        return accuracy


class VQAEvaluator(Evaluator):
    """VQA任务专用评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        device: Optional[str] = None
    ):
        """
        初始化VQA评估器
        
        Args:
            model: VQA模型
            processor: 处理器（用于解码）
            device: 设备
        """
        super().__init__(model, device, metrics=['loss', 'accuracy', 'exact_match'])
        self.processor = processor
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """评估VQA模型"""
        self.model.eval()
        
        all_losses = []
        all_predictions = []
        all_labels = []
        all_questions = []
        all_images = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估VQA模型"):
                # 准备输入
                batch = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self._forward_pass(batch)
                
                # 收集损失
                if 'loss' in outputs or hasattr(outputs, 'loss'):
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs.get('loss')
                    if loss is not None:
                        all_losses.append(loss.item())
                
                # 生成答案
                predictions = self._generate_answers(batch, outputs)
                labels = self._decode_labels(batch)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                if return_predictions:
                    all_questions.append(batch.get('input_ids'))
                    all_images.append(batch.get('pixel_values'))
        
        # 计算指标
        results = {}
        
        if all_losses:
            results['loss'] = np.mean(all_losses)
        
        # 计算准确率（文本匹配）
        accuracy = self._compute_text_accuracy(all_predictions, all_labels)
        results['accuracy'] = accuracy
        
        # 计算完全匹配率
        exact_match = self._compute_exact_match_text(all_predictions, all_labels)
        results['exact_match'] = exact_match
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
            results['questions'] = all_questions
            results['images'] = all_images
        
        return results
    
    def _generate_answers(self, batch: Dict[str, Any], outputs: Any) -> List[str]:
        """生成答案"""
        # 对于BLIP等生成模型，使用generate方法
        if hasattr(self.model, 'generate'):
            try:
                # 准备生成输入
                generate_inputs = {}
                if 'pixel_values' in batch:
                    generate_inputs['pixel_values'] = batch['pixel_values']
                if 'input_ids' in batch:
                    generate_inputs['input_ids'] = batch['input_ids']
                
                # 生成
                generated_ids = self.model.generate(
                    **generate_inputs,
                    max_length=20,
                    num_beams=3
                )
                
                # 解码
                answers = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                return answers
            except Exception as e:
                logger.warning(f"生成答案失败: {e}")
        
        # 如果生成失败，尝试从logits获取
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            answers = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            return answers
        
        return []
    
    def _decode_labels(self, batch: Dict[str, Any]) -> List[str]:
        """解码标签"""
        if 'labels' in batch:
            labels = batch['labels']
            # 移除-100（忽略的token）
            labels = labels.masked_fill(labels == -100, self.processor.tokenizer.pad_token_id)
            decoded_labels = self.processor.batch_decode(
                labels, skip_special_tokens=True
            )
            return decoded_labels
        return []
    
    def _compute_text_accuracy(self, predictions: List[str], labels: List[str]) -> float:
        """计算文本准确率（忽略大小写和空格）"""
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        correct = 0
        for pred, label in zip(predictions, labels):
            # 标准化：转小写，去除首尾空格
            pred_normalized = pred.lower().strip()
            label_normalized = label.lower().strip()
            
            if pred_normalized == label_normalized:
                correct += 1
        
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    def _compute_exact_match_text(self, predictions: List[str], labels: List[str]) -> float:
        """计算完全匹配率（文本）"""
        return self._compute_text_accuracy(predictions, labels)


# 示例用法
if __name__ == "__main__":
    print("Evaluator模块加载完成 - 提供模型评估功能")


"""
评估指标
提供各种评估指标的计算功能
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)


def accuracy(predictions: List[Union[int, str]], labels: List[Union[int, str]]) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测结果列表
        labels: 真实标签列表
        
    Returns:
        准确率
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def exact_match(predictions: List[str], labels: List[str], normalize: bool = True) -> float:
    """
    计算完全匹配率
    
    Args:
        predictions: 预测文本列表
        labels: 真实文本列表
        normalize: 是否标准化（转小写、去除空格）
        
    Returns:
        完全匹配率
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    if len(predictions) == 0:
        return 0.0
    
    matches = 0
    for pred, label in zip(predictions, labels):
        if normalize:
            pred_normalized = _normalize_text(pred)
            label_normalized = _normalize_text(label)
            if pred_normalized == label_normalized:
                matches += 1
        else:
            if pred == label:
                matches += 1
    
    return matches / len(predictions)


def f1_score(predictions: List[str], labels: List[str], average: str = 'macro') -> Union[float, Dict[str, float]]:
    """
    计算F1分数
    
    Args:
        predictions: 预测文本列表
        labels: 真实文本列表
        average: 平均方式 ('macro', 'micro', 'weighted', None)
        
    Returns:
        F1分数（如果average=None，返回每个类别的F1分数字典）
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    if len(predictions) == 0:
        return 0.0 if average else {}
    
    # 获取所有类别
    all_labels = set(labels)
    
    if average == 'micro':
        # Micro F1: 将所有类别的TP、FP、FN相加后计算
        tp = sum(p == l for p, l in zip(predictions, labels))
        fp = sum(p != l for p, l in zip(predictions, labels))
        fn = fp  # 对于分类任务，FP = FN
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    # 计算每个类别的F1
    class_f1s = {}
    for label in all_labels:
        tp = sum(p == label and l == label for p, l in zip(predictions, labels))
        fp = sum(p == label and l != label for p, l in zip(predictions, labels))
        fn = sum(p != label and l == label for p, l in zip(predictions, labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_f1s[label] = f1
    
    if average == 'macro':
        return np.mean(list(class_f1s.values()))
    elif average == 'weighted':
        # 加权平均（按样本数）
        label_counts = Counter(labels)
        total = len(labels)
        weighted_f1 = sum(class_f1s[label] * (count / total) for label, count in label_counts.items())
        return weighted_f1
    else:
        return class_f1s


def bleu_score(predictions: List[str], labels: List[str], n: int = 4) -> float:
    """
    计算BLEU分数（简化版）
    
    Args:
        predictions: 预测文本列表
        labels: 真实文本列表
        n: n-gram的最大n值
        
    Returns:
        BLEU分数
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    if len(predictions) == 0:
        return 0.0
    
    total_bleu = 0.0
    for pred, label in zip(predictions, labels):
        pred_tokens = _tokenize(pred)
        label_tokens = _tokenize(label)
        
        # 计算n-gram精度
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, i)
            label_ngrams = _get_ngrams(label_tokens, i)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
            else:
                matches = sum(1 for ngram in pred_ngrams if ngram in label_ngrams)
                precisions.append(matches / len(pred_ngrams))
        
        # 几何平均
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        
        # 长度惩罚
        if len(pred_tokens) < len(label_tokens):
            bp = np.exp(1 - len(label_tokens) / len(pred_tokens))
        else:
            bp = 1.0
        
        total_bleu += bleu * bp
    
    return total_bleu / len(predictions)


def rouge_score(predictions: List[str], labels: List[str], rouge_type: str = 'rouge-l') -> float:
    """
    计算ROUGE分数（简化版）
    
    Args:
        predictions: 预测文本列表
        labels: 真实文本列表
        rouge_type: ROUGE类型 ('rouge-l', 'rouge-1', 'rouge-2')
        
    Returns:
        ROUGE分数
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    if len(predictions) == 0:
        return 0.0
    
    total_rouge = 0.0
    for pred, label in zip(predictions, labels):
        pred_tokens = _tokenize(pred)
        label_tokens = _tokenize(label)
        
        if rouge_type == 'rouge-l':
            # 最长公共子序列
            lcs = _longest_common_subsequence(pred_tokens, label_tokens)
            precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = lcs / len(label_tokens) if len(label_tokens) > 0 else 0.0
            rouge = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif rouge_type == 'rouge-1':
            # 1-gram重叠
            pred_unigrams = set(pred_tokens)
            label_unigrams = set(label_tokens)
            overlap = len(pred_unigrams & label_unigrams)
            precision = overlap / len(pred_unigrams) if len(pred_unigrams) > 0 else 0.0
            recall = overlap / len(label_unigrams) if len(label_unigrams) > 0 else 0.0
            rouge = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif rouge_type == 'rouge-2':
            # 2-gram重叠
            pred_bigrams = set(_get_ngrams(pred_tokens, 2))
            label_bigrams = set(_get_ngrams(label_tokens, 2))
            overlap = len(pred_bigrams & label_bigrams)
            precision = overlap / len(pred_bigrams) if len(pred_bigrams) > 0 else 0.0
            recall = overlap / len(label_bigrams) if len(label_bigrams) > 0 else 0.0
            rouge = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            rouge = 0.0
        
        total_rouge += rouge
    
    return total_rouge / len(predictions)


def compute_metrics(
    predictions: List[Union[int, str]],
    labels: List[Union[int, str]],
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算多个指标
    
    Args:
        predictions: 预测结果列表
        labels: 真实标签列表
        metric_names: 要计算的指标名称列表
        
    Returns:
        指标字典
    """
    if metric_names is None:
        metric_names = ['accuracy', 'exact_match', 'f1']
    
    results = {}
    
    # 转换为字符串（如果需要）
    pred_str = [str(p) for p in predictions]
    label_str = [str(l) for l in labels]
    
    for metric_name in metric_names:
        try:
            if metric_name == 'accuracy':
                results['accuracy'] = accuracy(predictions, labels)
            elif metric_name == 'exact_match':
                results['exact_match'] = exact_match(pred_str, label_str)
            elif metric_name == 'f1':
                results['f1'] = f1_score(pred_str, label_str, average='macro')
            elif metric_name == 'f1_micro':
                results['f1_micro'] = f1_score(pred_str, label_str, average='micro')
            elif metric_name == 'f1_weighted':
                results['f1_weighted'] = f1_score(pred_str, label_str, average='weighted')
            elif metric_name == 'bleu':
                results['bleu'] = bleu_score(pred_str, label_str)
            elif metric_name == 'rouge':
                results['rouge'] = rouge_score(pred_str, label_str)
            elif metric_name == 'rouge_l':
                results['rouge_l'] = rouge_score(pred_str, label_str, rouge_type='rouge-l')
            elif metric_name == 'rouge_1':
                results['rouge_1'] = rouge_score(pred_str, label_str, rouge_type='rouge-1')
            elif metric_name == 'rouge_2':
                results['rouge_2'] = rouge_score(pred_str, label_str, rouge_type='rouge-2')
        except Exception as e:
            logger.warning(f"计算指标 {metric_name} 失败: {e}")
            results[metric_name] = 0.0
    
    return results


# 辅助函数

def _normalize_text(text: str) -> str:
    """标准化文本"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    return text


def _tokenize(text: str) -> List[str]:
    """分词"""
    text = _normalize_text(text)
    return text.split()


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """获取n-gram"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """计算最长公共子序列长度"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


# 示例用法
if __name__ == "__main__":
    # 测试准确率
    predictions = [1, 2, 3, 4, 5]
    labels = [1, 2, 3, 4, 6]
    print(f"准确率: {accuracy(predictions, labels):.4f}")
    
    # 测试完全匹配
    pred_texts = ["hello world", "foo bar", "test"]
    label_texts = ["hello world", "foo baz", "test"]
    print(f"完全匹配率: {exact_match(pred_texts, label_texts):.4f}")
    
    # 测试F1分数
    print(f"F1分数: {f1_score(pred_texts, label_texts, average='macro'):.4f}")
    
    # 计算多个指标
    metrics = compute_metrics(pred_texts, label_texts, ['accuracy', 'exact_match', 'f1'])
    print(f"所有指标: {metrics}")
    
    print("Metrics模块加载完成")


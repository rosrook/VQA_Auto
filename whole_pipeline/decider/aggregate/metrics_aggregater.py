"""
指标聚合器 (metrics_aggrefater.py)

该模块负责对从多个评估结果中提取的指标进行语义去重和聚合,
将相似的指标合并,构建统一的指标体系。

理论依据:
1. 语义相似度计算 (Semantic Similarity):
   - 基于文本嵌入的语义相似度度量
   - 参考 Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
   - 使用 TF-IDF 向量化和余弦相似度作为备选方案

2. 文本聚类 (Text Clustering):
   - 通过相似度阈值进行层次聚类
   - 合并语义相近的指标描述

3. 指标体系构建:
   - 将通用的 General Metric 与特定的 Specific Metric 进行关联
   - 构建层次化的指标结构
"""

import argparse
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 尝试导入可选的语义相似度库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple text similarity")

import requests
import time


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class MetricItem:
    """单个指标项"""
    general_metric_description: str
    specific_metric_description: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "general_metric_description": self.general_metric_description,
            "specific_metric_description": self.specific_metric_description
        }
    
    def combined_text(self) -> str:
        """获取组合文本用于相似度计算"""
        return f"{self.general_metric_description} {self.specific_metric_description}"


# --------------------------------------------------------------------------- #
# Embedding API 客户端
# --------------------------------------------------------------------------- #
class EmbeddingAPIClient:
    """
    Embedding API 客户端,支持多种 embedding API 服务
    - OpenAI API
    - HuggingFace Inference API
    - 自定义兼容 OpenAI 格式的 API
    """
    
    def __init__(
        self,
        api_type: str = "openai",  # "openai", "hf", "custom"
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2
    ):
        """
        初始化 Embedding API 客户端
        
        参数:
            api_type: API 类型 ("openai", "hf", "custom")
            api_key: API 密钥
            base_url: API 基础 URL (OpenAI: https://api.openai.com/v1, HF: https://api-inference.huggingface.co)
            model: 模型名称 (OpenAI: "text-embedding-3-small", HF: "sentence-transformers/all-MiniLM-L6-v2")
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
        """
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = base_url or self._get_default_url(api_type)
        self.model = model or self._get_default_model(api_type)
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"初始化 Embedding API 客户端: type={api_type}, model={self.model}")
    
    def _get_default_url(self, api_type: str) -> str:
        """获取默认 API URL"""
        defaults = {
            "openai": "https://api.openai.com/v1",
            "hf": "https://api-inference.huggingface.co",
            "custom": "https://api.openai.com/v1"  # 假设兼容 OpenAI 格式
        }
        return defaults.get(api_type, defaults["openai"])
    
    def _get_default_model(self, api_type: str) -> str:
        """获取默认模型名称"""
        defaults = {
            "openai": "text-embedding-3-small",
            "hf": "sentence-transformers/all-MiniLM-L6-v2",
            "custom": "text-embedding-3-small"
        }
        return defaults.get(api_type, defaults["openai"])
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        获取文本的 embedding 向量
        
        参数:
            texts: 文本列表
            
        返回:
            numpy.ndarray: embedding 向量数组, shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self.api_type == "openai":
            return self._encode_openai(texts)
        elif self.api_type == "hf":
            return self._encode_hf(texts)
        else:  # custom, 假设兼容 OpenAI 格式
            return self._encode_openai(texts)
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """
        使用 OpenAI API 获取 embedding (批量调用)
        OpenAI API 支持一次传入多个文本，大幅提升性能
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenAI API 支持批量输入，一次最多处理 2048 个文本
        # 批量处理可以大幅减少 API 调用次数
        batch_size = 100  # 每批处理100个文本，避免单次请求过大
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            payload = {
                "input": batch_texts,  # 批量输入
                "model": self.model
            }
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()
                    # 批量返回的格式: {"data": [{"embedding": [...]}, ...], ...}
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt
                        logger.warning(f"Embedding API 批量请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}, {wait_time}秒后重试")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Embedding API 批量请求最终失败: {e}")
                        raise
        
        return np.array(all_embeddings)
    
    def _encode_hf(self, texts: List[str]) -> np.ndarray:
        """使用 HuggingFace Inference API 获取 embedding"""
        url = f"{self.base_url}/pipeline/feature-extraction/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                embeddings_list = response.json()
                return np.array(embeddings_list)
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"HF Embedding API 请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}, {wait_time}秒后重试")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HF Embedding API 请求最终失败: {e}")
                    raise


class MetricsAggregator:
    """
    指标聚合器: 对指标列表进行语义去重/聚合
    - 支持 embedding / TF-IDF / 简单 Jaccard 回退
    - General 优先: General 相似度高时合并, Specific 差异会被保留到列表
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        general_similarity_threshold: float = 0.80,
        use_embedding: bool = True,
        embedding_api_config: Optional[Dict] = None
    ):
        """
        初始化聚合器
        
        参数:
            similarity_threshold: 综合相似度阈值,超过此值则认为是相似指标 (0-1)
            general_similarity_threshold: General metric 相似度阈值,如果 General 相似度超过此值,
                                       即使 Specific 不同也会合并到同一个 general 下 (0-1)
            use_embedding: 是否使用 embedding API 计算相似度
            embedding_api_config: Embedding API 配置字典,格式:
                {
                    "api_type": "openai" | "hf" | "custom",
                    "api_key": "...",
                    "base_url": "...",  # 可选
                    "model": "...",     # 可选
                    "timeout": 30,      # 可选
                    "max_retries": 2    # 可选
                }
                如果为 None 且 use_embedding=True, 将尝试从环境变量读取
        """
        self.similarity_threshold = similarity_threshold
        self.general_similarity_threshold = general_similarity_threshold
        self.use_embedding = use_embedding
        
        # 初始化 embedding 缓存（无论是否使用 embedding 都初始化，避免属性错误）
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # 初始化 embedding API 客户端
        self.embedding_client = None
        if use_embedding:
            try:
                if embedding_api_config:
                    self.embedding_client = EmbeddingAPIClient(**embedding_api_config)
                else:
                    # 尝试从环境变量读取配置
                    import os
                    api_key = os.getenv("EMBEDDING_API_KEY")
                    api_type = os.getenv("EMBEDDING_API_TYPE", "openai")
                    base_url = os.getenv("EMBEDDING_BASE_URL")
                    model = os.getenv("EMBEDDING_MODEL")
                    
                    if api_key:
                        self.embedding_client = EmbeddingAPIClient(
                            api_type=api_type,
                            api_key=api_key,
                            base_url=base_url,
                            model=model
                        )
                    else:
                        logger.warning("未提供 embedding API 配置且环境变量未设置, 使用 TF-IDF")
                        self.use_embedding = False
            except Exception as e:
                logger.warning(f"无法初始化 embedding API 客户端: {e}, 使用 TF-IDF")
                self.use_embedding = False
        
        # 初始化 TF-IDF vectorizer (作为备选，始终初始化避免属性错误)
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            if not self.use_embedding:
                logger.info("使用 TF-IDF 进行相似度计算")
        
        logger.info(
            f"初始化指标聚合器,综合相似度阈值: {similarity_threshold}, "
            f"General相似度阈值: {general_similarity_threshold}, "
            f"使用Embedding API: {self.use_embedding}"
        )
    
    def _calculate_text_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        计算两个文本的相似度
        
        参数:
            text1: 第一个文本
            text2: 第二个文本
            
        返回:
            相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 使用 embedding API (带缓存)
        if self.use_embedding and self.embedding_client:
            try:
                # 检查缓存
                vec1 = self._embedding_cache.get(text1)
                vec2 = self._embedding_cache.get(text2)
                
                # 批量获取缺失的 embedding
                texts_to_encode = []
                if vec1 is None:
                    texts_to_encode.append(text1)
                if vec2 is None:
                    texts_to_encode.append(text2)
                
                if texts_to_encode:
                    new_embeddings = self.embedding_client.encode(texts_to_encode)
                    # 更新缓存并赋值
                    idx = 0
                    if vec1 is None:
                        vec1 = new_embeddings[idx]
                        self._embedding_cache[text1] = vec1
                        idx += 1
                    if vec2 is None:
                        vec2 = new_embeddings[idx]
                        self._embedding_cache[text2] = vec2
                else:
                    # 都从缓存获取
                    vec1 = self._embedding_cache[text1]
                    vec2 = self._embedding_cache[text2]
                
                # 计算余弦相似度
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity(
                        vec1.reshape(1, -1),
                        vec2.reshape(1, -1)
                    )[0][0]
                else:
                    # 使用 numpy 计算余弦相似度
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = dot_product / (norm1 * norm2)
                return float(similarity)
            except Exception as e:
                logger.warning(f"Embedding API 计算失败: {e}, 回退到简单方法")
        
        # 使用 TF-IDF
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None:
            try:
                vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return float(similarity)
            except Exception as e:
                logger.warning(f"TF-IDF 计算失败: {e}, 使用简单方法")
        
        # 简单的文本相似度 (Jaccard 相似度)
        return self._simple_text_similarity(text1, text2)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """
        简单的文本相似度计算 (Jaccard 相似度)
        
        参数:
            text1: 第一个文本
            text2: 第二个文本
            
        返回:
            相似度分数 (0-1)
        """
        # 转换为小写并分词
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard 相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _are_metrics_similar(
        self,
        metric1: MetricItem,
        metric2: MetricItem
    ) -> Tuple[bool, float]:
        """
        判断两个指标是否相似
        
        优先检查 General metric 相似度: 如果 General 相似度高,则合并,
        即使 Specific 不同,也会将不同的 Specific 归到同一个 General 下。
        
        参数:
            metric1: 第一个指标
            metric2: 第二个指标
            
        返回:
            (是否相似, 相似度分数)
        """
        # 首先计算 General 相似度 (优先级最高)
        general_similarity = self._calculate_text_similarity(
            metric1.general_metric_description,
            metric2.general_metric_description
        )
        
        # 如果 General 相似度足够高,直接返回 True (应该合并)
        # 即使 Specific 不同,也会将它们归到同一个 general 下
        if general_similarity >= self.general_similarity_threshold:
            return True, general_similarity
        
        # 如果 General 相似度不够高,再检查综合相似度
        combined1 = metric1.combined_text()
        combined2 = metric2.combined_text()
        
        # 组合相似度 (General 和 Specific 都考虑)
        combined_similarity = self._calculate_text_similarity(combined1, combined2)
        
        # Specific 相似度
        specific_similarity = self._calculate_text_similarity(
            metric1.specific_metric_description,
            metric2.specific_metric_description
        )
        
        # 综合相似度: General 权重 0.6, Specific 权重 0.2, Combined 权重 0.2
        weighted_similarity = (
            general_similarity * 0.6 +
            specific_similarity * 0.2 +
            combined_similarity * 0.2
        )
        
        is_similar = weighted_similarity >= self.similarity_threshold
        
        return is_similar, weighted_similarity
    
    # 这里暂时选择保留最长的字符串，因为合并时最长的往往信息量最大，以后升级时可以选择计算信息熵
    def _merge_metric_descriptions(
        self,
        descriptions: List[str]
    ) -> str:
        """
        合并多个描述文本,选择最详细或最具代表性的描述
        
        参数:
            descriptions: 描述文本列表
            
        返回:
            合并后的描述
        """
        if not descriptions:
            return ""
        
        # 去除空字符串
        descriptions = [d.strip() for d in descriptions if d.strip()]
        
        if not descriptions:
            return ""
        
        # 选择最长的描述(通常更详细)
        merged = max(descriptions, key=len)
        
        return merged
    
    def deduplicate_metrics(
        self,
        metrics: List[Dict[str, str]]
    ) -> List[Dict[str, List[str]]]:
        """
        对指标列表进行语义去重
        
        参数:
            metrics: 指标列表,每个指标包含:
                - "General metric description" 或 "general_metric_description"
                - "Specific metric description" 或 "specific_metric_description"
                
        返回:
            List[Dict]: 去重后的指标列表,格式为:
                [{"general": str, "specific": List[str]}, ...]
        """
        logger.info(f"开始对 {len(metrics)} 个指标进行去重")
        
        # 转换为 MetricItem 列表
        metric_items = []
        for metric in metrics:
            # 处理字段名的不同格式
            general = metric.get("General metric description") or metric.get("general_metric_description") or ""
            specific = metric.get("Specific metric description") or metric.get("specific_metric_description") or ""
            
            if not general and not specific:
                logger.warning(f"跳过无效指标: {metric}")
                continue
            
            metric_items.append(MetricItem(
                general_metric_description=general,
                specific_metric_description=specific
            ))
        
        logger.info(f"有效指标数量: {len(metric_items)}")
        
        # 使用聚类方法进行去重
        merged_groups = []
        used_indices = set()
        
        for i, metric1 in enumerate(metric_items):
            if i in used_indices:
                continue
            
            # 创建新的合并组
            current_group = {
                "general": metric1.general_metric_description,
                "specific": [metric1.specific_metric_description] if metric1.specific_metric_description else []
            }
            
            used_indices.add(i)
            
            # 查找相似的指标
            for j, metric2 in enumerate(metric_items[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                is_similar, similarity = self._are_metrics_similar(metric1, metric2)
                
                if is_similar:
                    # 检查 General 相似度,判断合并原因
                    general_sim = self._calculate_text_similarity(
                        metric1.general_metric_description,
                        metric2.general_metric_description
                    )
                    merge_reason = "General 相似" if general_sim >= self.general_similarity_threshold else "综合相似"
                    
                    # 合并 General description (选择更详细的)
                    general_options = [
                        current_group["general"],
                        metric2.general_metric_description
                    ]
                    merged_general = self._merge_metric_descriptions(general_options)
                    
                    # 添加 Specific description (如果不同)
                    specific_added = False
                    if metric2.specific_metric_description:
                        if metric2.specific_metric_description not in current_group["specific"]:
                            current_group["specific"].append(metric2.specific_metric_description)
                            specific_added = True
                    
                    logger.debug(
                        f"合并相似指标 ({merge_reason}, 相似度: {similarity:.3f}):\n"
                        f"  General: '{metric1.general_metric_description}' <-> '{metric2.general_metric_description}'\n"
                        f"  合并后 General: '{merged_general}'\n"
                        f"  Specific: '{metric1.specific_metric_description}' <-> '{metric2.specific_metric_description}'\n"
                        f"  (General相似度: {general_sim:.3f}, 当前specific列表包含: {len(current_group['specific'])} 项"
                        f"{', 新增specific' if specific_added else ', specific已存在'})"
                    )
                    
                    current_group["general"] = merged_general
                    
                    used_indices.add(j)
            
            merged_groups.append(current_group)
        
        logger.info(f"去重完成: {len(metric_items)} -> {len(merged_groups)} 个指标")
        
        return merged_groups
    
    def load_metrics_from_folder(
        self,
        folder_path: str,
        file_pattern: str = "*.json"
    ) -> List[Dict[str, str]]:
        """
        从文件夹中读取指标数据 (预留功能,待实现)
        
        参数:
            folder_path: 数据文件夹路径
            file_pattern: 文件匹配模式
            
        返回:
            List[Dict]: 指标列表
            
        注意:
            此功能暂时未实现,等待后续补充
        """
        logger.warning("load_metrics_from_folder 功能尚未实现,待后续补充")
        raise NotImplementedError("此功能尚未实现,等待后续补充")
        
        # TODO: 实现从文件夹读取指标数据的功能
        # 1. 遍历文件夹中的所有匹配文件
        # 2. 解析 JSON 文件中的指标数据
        # 3. 提取 "recommended_metrics" 字段
        # 4. 返回统一的指标列表格式
        pass
    
    def save_merged_metrics(
        self,
        merged_metrics: List[Dict[str, List[str]]],
        output_path: str
    ):
        """
        保存合并后的指标到文件
        
        参数:
            merged_metrics: 合并后的指标列表
            output_path: 输出文件路径
        """
        logger.info(f"保存合并后的指标到 {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_metrics, f, ensure_ascii=False, indent=2)
        
        logger.info("保存完成")


@dataclass
class ProcessingResult:
    """处理结果"""
    hard_cases: List[Dict]  # 难例列表（保留原格式）
    merged_metrics: List[Dict[str, List[str]]]  # 合并去重后的指标
    deduplication_stats: Dict  # 去重统计信息（重复度）
    difficulty_scores: Dict[str, float]  # 每个样本的难度分数
    confidence_scores: Dict[str, float]  # 每个样本的置信度分数


class MultiModelMetricsProcessor:
    """
    多模型指标处理器 (核心流水线)
    步骤:
      1) 评分分歧 -> 难度/置信度，分出难例
      2) 过滤 metrics: SOTA 保留，差异过大丢弃其余模型的 metrics
      3) 样本内去重 -> 单样本 merged_metrics
      4) (可选) 跨样本宽松去重 -> 全局指标
      5) 生成决策 prompt (包含 agents 指南)
    """
    
    def __init__(
        self,
        difficulty_threshold: float = 3.0,
        score_diff_threshold: float = 2.5,
        similarity_threshold: float = 0.75,
        general_similarity_threshold: float = 0.80,
        use_embedding: bool = True,
        embedding_api_config: Optional[Dict] = None
    ):
        """
        初始化处理器
        
        参数:
            difficulty_threshold: 难度阈值,打分标准差超过此值认为是难例
            score_diff_threshold: 打分差异阈值,与SOTA差异超过此值则抛弃该模型的metrics
            similarity_threshold: 指标相似度阈值（用于去重）
            general_similarity_threshold: General指标相似度阈值（用于去重）
            use_embedding: 是否使用embedding API计算相似度
            embedding_api_config: Embedding API 配置字典,格式见 MetricsAggregator.__init__
        """
        self.difficulty_threshold = difficulty_threshold
        self.score_diff_threshold = score_diff_threshold
        
        # 初始化指标聚合器（用于去重）
        self.metrics_aggregator = MetricsAggregator(
            similarity_threshold=similarity_threshold,
            general_similarity_threshold=general_similarity_threshold,
            use_embedding=use_embedding,
            embedding_api_config=embedding_api_config
        )
        
        logger.info(
            f"初始化多模型指标处理器: "
            f"难度阈值={difficulty_threshold}, "
            f"打分差异阈值={score_diff_threshold}"
        )
    
    def _calculate_disagreement(
        self,
        scores: List[float]
    ) -> Tuple[float, float]:
        """
        计算模型间分歧度
        
        参数:
            scores: 评分列表 [weak_score, normal_score, sota_score]
            
        返回:
            (标准差, 难度分数)
        """
        if not scores or len(scores) < 2:
            return 0.0, 0.0
        
        std_score = np.std(scores)
        max_score = max(scores)
        min_score = min(scores)
        range_score = max_score - min_score
        
        # 难度分数：综合考虑标准差和极差
        difficulty_score = (std_score * 0.7 + range_score * 0.3)
        
        return std_score, difficulty_score
    
    def _calculate_confidence(
        self,
        correctness_scores: List[float],
        training_scores: List[float],
        correctness_confidences: List[float],
        training_confidences: List[float]
    ) -> float:
        """
        计算综合置信度
        
        参数:
            correctness_scores: 正确性评分列表
            training_scores: 训练质量评分列表
            correctness_confidences: 正确性置信度列表
            training_confidences: 训练质量置信度列表
            
        返回:
            综合置信度 (0-1)
        """
        # 计算评分一致性
        correctness_std = np.std(correctness_scores) if len(correctness_scores) > 1 else 0.0
        training_std = np.std(training_scores) if len(training_scores) > 1 else 0.0
        
        # 标准差越小,一致性越高
        correctness_consistency = 1.0 / (1.0 + correctness_std)
        training_consistency = 1.0 / (1.0 + training_std)
        
        # 平均模型置信度
        avg_correctness_confidence = np.mean(correctness_confidences) if correctness_confidences else 0.5
        avg_training_confidence = np.mean(training_confidences) if training_confidences else 0.5
        
        # 综合置信度
        overall_confidence = (
            correctness_consistency * 0.3 +
            training_consistency * 0.3 +
            avg_correctness_confidence * 0.2 +
            avg_training_confidence * 0.2
        )
        
        return float(np.clip(overall_confidence, 0.0, 1.0))
    
    def _is_hard_case(
        self,
        difficulty_score: float,
        correctness_std: float,
        training_std: float
    ) -> bool:
        """
        判断是否为难例
        
        参数:
            difficulty_score: 难度分数
            correctness_std: 正确性评分标准差
            training_std: 训练质量评分标准差
            
        返回:
            是否为难例
        """
        # 如果难度分数或任一维度的标准差超过阈值,认为是难例
        max_std = max(correctness_std, training_std)
        return difficulty_score >= self.difficulty_threshold or max_std >= self.difficulty_threshold
    
    def _should_keep_metrics(
        self,
        model_score: float,
        sota_score: float,
        model_name: str
    ) -> bool:
        """
        判断是否应该保留某个模型的metrics
        
        参数:
            model_score: 当前模型的评分
            sota_score: SOTA模型的评分
            model_name: 模型名称
            
        返回:
            是否应该保留
        """
        # SOTA模型的metrics始终保留
        if model_name == "sota":
            return True
        
        # 计算与SOTA的差异
        score_diff = abs(model_score - sota_score)
        
        # 如果差异超过阈值,不保留该模型的metrics
        return score_diff < self.score_diff_threshold
    
    def process_single_sample(
        self,
        sample_id: str,
        weak_result: Dict,
        normal_result: Dict,
        sota_result: Dict
    ) -> Dict:
        """
        处理单个样本的三个模型结果
        
        参数:
            sample_id: 样本ID
            weak_result: weak模型的结果
            normal_result: normal模型的结果
            sota_result: sota模型的结果
            
        返回:
            处理结果字典
        """
        # 提取评分
        correctness_scores = [
            weak_result.get("correctness_score", 0.0),
            normal_result.get("correctness_score", 0.0),
            sota_result.get("correctness_score", 0.0)
        ]
        
        training_scores = [
            weak_result.get("training_quality_score", 0.0),
            normal_result.get("training_quality_score", 0.0),
            sota_result.get("training_quality_score", 0.0)
        ]
        
        correctness_confidences = [
            weak_result.get("correctness_confidence", 0.5),
            normal_result.get("correctness_confidence", 0.5),
            sota_result.get("correctness_confidence", 0.5)
        ]
        
        training_confidences = [
            weak_result.get("training_quality_confidence", 0.5),
            normal_result.get("training_quality_confidence", 0.5),
            sota_result.get("training_quality_confidence", 0.5)
        ]
        
        # 计算分歧度和难度
        correctness_std, correctness_difficulty = self._calculate_disagreement(correctness_scores)
        training_std, training_difficulty = self._calculate_disagreement(training_scores)
        overall_difficulty = (correctness_difficulty + training_difficulty) / 2.0
        
        # 计算置信度
        confidence = self._calculate_confidence(
            correctness_scores,
            training_scores,
            correctness_confidences,
            training_confidences
        )
        
        # 判断是否为难例
        is_hard = self._is_hard_case(overall_difficulty, correctness_std, training_std)
        
        # 收集metrics（根据规则筛选）
        all_metrics = []
        results = {
            "weak": weak_result,
            "normal": normal_result,
            "sota": sota_result
        }
        
        # 综合评分（用于判断是否保留metrics）
        combined_scores = [
            (correctness_scores[i] + training_scores[i]) / 2.0
            for i in range(3)
        ]
        sota_combined_score = combined_scores[2]  # SOTA在第3个位置
        
        for i, (model_name, result) in enumerate(results.items()):
            model_combined_score = combined_scores[i]
            
            # 判断是否应该保留该模型的metrics
            if self._should_keep_metrics(model_combined_score, sota_combined_score, model_name):
                metrics = result.get("recommended_metrics", [])
                for metric in metrics:
                    # 转换为标准格式
                    metric_dict = {
                        "General metric description": metric.get("general_metric_description", ""),
                        "Specific metric description": metric.get("specific_metric_description", "")
                    }
                    if metric_dict["General metric description"] or metric_dict["Specific metric description"]:
                        all_metrics.append(metric_dict)
                        logger.debug(f"保留 {model_name} 模型的metric: {metric_dict['General metric description']}")
            else:
                logger.debug(
                    f"抛弃 {model_name} 模型的metrics "
                    f"(与SOTA差异: {abs(model_combined_score - sota_combined_score):.2f})"
                )
        
        return {
            "sample_id": sample_id,
            "is_hard_case": is_hard,
            "difficulty_score": overall_difficulty,
            "confidence_score": confidence,
            "correctness_std": correctness_std,
            "training_std": training_std,
            "metrics": all_metrics,
            "raw_results": {
                "weak": weak_result,
                "normal": normal_result,
                "sota": sota_result
            }
        }
    
    def process_batch(
        self,
        batch_data: List[Dict[str, Dict]]
    ) -> ProcessingResult:
        """
        批量处理样本
        
        参数:
            batch_data: 批量数据,格式为:
                [
                    {
                        "sample_id": "sample_1",
                        "weak": {...},
                        "normal": {...},
                        "sota": {...}
                    },
                    ...
                ]
                
        返回:
            ProcessingResult: 处理结果
        """
        logger.info(f"开始批量处理 {len(batch_data)} 个样本")
        
        hard_cases = []
        all_metrics = []
        difficulty_scores = {}
        confidence_scores = {}
        
        # 处理每个样本
        for item in batch_data:
            sample_id = item.get("sample_id", "unknown")
            weak_result = item.get("weak", {})
            normal_result = item.get("normal", {})
            sota_result = item.get("sota", {})
            
            # 处理单个样本
            result = self.process_single_sample(
                sample_id, weak_result, normal_result, sota_result
            )
            
            # 记录难度和置信度
            difficulty_scores[sample_id] = result["difficulty_score"]
            confidence_scores[sample_id] = result["confidence_score"]
            
            # 如果是难例,保留完整结果
            if result["is_hard_case"]:
                hard_cases.append({
                    "sample_id": sample_id,
                    "difficulty_score": result["difficulty_score"],
                    "confidence_score": result["confidence_score"],
                    "weak": result["raw_results"]["weak"],
                    "normal": result["raw_results"]["normal"],
                    "sota": result["raw_results"]["sota"]
                })
                logger.info(f"样本 {sample_id} 判定为难例 (难度: {result['difficulty_score']:.2f})")
            else:
                # 非难例,收集metrics
                all_metrics.extend(result["metrics"])
        
        logger.info(f"难例数量: {len(hard_cases)}, 收集到 {len(all_metrics)} 个metrics")
        
        # 对metrics进行去重
        logger.info("开始对metrics进行去重...")
        merged_metrics = self.metrics_aggregator.deduplicate_metrics(all_metrics)
        
        # 计算去重统计信息（重复度）
        original_count = len(all_metrics)
        merged_count = len(merged_metrics)
        total_specific_before = original_count  # 原始每个metric都有一个specific
        total_specific_after = sum(len(m.get("specific", [])) for m in merged_metrics)
        
        # 计算重复度
        deduplication_rate = 1.0 - (merged_count / original_count) if original_count > 0 else 0.0
        specific_reduction_rate = 1.0 - (total_specific_after / total_specific_before) if total_specific_before > 0 else 0.0
        
        deduplication_stats = {
            "original_metric_count": original_count,
            "merged_metric_count": merged_count,
            "deduplication_rate": deduplication_rate,  # 重复度（去重比例）
            "reduction_ratio": merged_count / original_count if original_count > 0 else 0.0,  # 压缩比
            "original_specific_count": total_specific_before,
            "merged_specific_count": total_specific_after,
            "specific_reduction_rate": specific_reduction_rate,
            "hard_cases_count": len(hard_cases)
        }
        
        logger.info(f"去重完成: {original_count} -> {merged_count} 个指标 (重复度: {deduplication_rate:.2%})")
        
        return ProcessingResult(
            hard_cases=hard_cases,
            merged_metrics=merged_metrics,
            deduplication_stats=deduplication_stats,
            difficulty_scores=difficulty_scores,
            confidence_scores=confidence_scores
        )

    def process_file_per_sample(
        self,
        file_path: str,
        include_hard_cases_in_samples: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        读取 JSON/JSONL 文件,对每个样本单独处理并返回:
        - samples_metrics: 每个样本去重聚合后的指标
        - hard_cases: 难例列表(保留三模型原始结果)
        - include_hard_cases_in_samples: 为 True 时,难例也会出现在 samples_metrics 中

        输入文件格式示例 (JSON list):
        [
          {
            "weak": {...},
            "normal": {...},
            "sota": {...}
          },
          ...
        ]
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取数据: 支持 .json (list) 或 .jsonl (每行一个 dict)
        if path.suffix.lower() == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as f:
                records = json.load(f)

        if not isinstance(records, list):
            raise ValueError("输入文件内容应为列表,每个元素包含 weak/normal/sota 结果")

        samples_metrics: List[Dict] = []
        hard_cases: List[Dict] = []

        for item in records:
            # 兼容字段名: 如果最外层就是三模型字典,尝试读取 sample_id
            sample_id = (
                item.get("sample_id")
                or item.get("weak", {}).get("sample_id")
                or item.get("normal", {}).get("sample_id")
                or item.get("sota", {}).get("sample_id")
                or "unknown"
            )

            weak_result = item.get("weak", {})
            normal_result = item.get("normal", {})
            sota_result = item.get("sota", {})

            result = self.process_single_sample(
                sample_id, weak_result, normal_result, sota_result
            )

            if result["is_hard_case"]:
                hard_case_item = {
                    "sample_id": sample_id,
                    "difficulty_score": result["difficulty_score"],
                    "confidence_score": result["confidence_score"],
                    "weak": result["raw_results"]["weak"],
                    "normal": result["raw_results"]["normal"],
                    "sota": result["raw_results"]["sota"],
                }
                hard_cases.append(hard_case_item)
                # 默认不把难例放入 samples_metrics; 若需要则继续处理
                if not include_hard_cases_in_samples:
                    continue

            # 对单个样本的 metrics 进行去重
            merged_metrics = self.metrics_aggregator.deduplicate_metrics(
                result["metrics"]
            )

            samples_metrics.append({
                "sample_id": sample_id,
                "merged_metrics": merged_metrics,
                "difficulty_score": result["difficulty_score"],
                "confidence_score": result["confidence_score"],
                "correctness_std": result["correctness_std"],
                "training_std": result["training_std"],
            })

        return samples_metrics, hard_cases

    # ----------------------------- 新增收尾功能 ----------------------------- #
    def aggregate_all_metrics_from_samples(
        self,
        samples_metrics: List[Dict],
        similarity_threshold: float = 0.65,
        general_similarity_threshold: float = 0.75
    ) -> List[Dict[str, List[str]]]:
        """
        将所有样本的metrics进行再次去重/聚合,生成跨样本的完整指标列表
        (阈值放宽,避免过度拆分相似指标)
        """
        # 收集所有metric对 (general, specific)
        collected: List[Dict[str, str]] = []
        for sm in samples_metrics:
            for metric in sm.get("merged_metrics", []):
                general = metric.get("general", "")
                specifics = metric.get("specific", []) or [""]
                for sp in specifics:
                    collected.append({
                        "General metric description": general,
                        "Specific metric description": sp,
                    })

        # 使用更宽松阈值的聚合器
        # 复用已有的 embedding 配置（如果有）
        embedding_config = None
        if self.metrics_aggregator.use_embedding and self.metrics_aggregator.embedding_client:
            # 从现有客户端提取配置以便复用
            client = self.metrics_aggregator.embedding_client
            embedding_config = {
                "api_type": client.api_type,
                "api_key": client.api_key,
                "base_url": client.base_url,
                "model": client.model,
                "timeout": client.timeout,
                "max_retries": client.max_retries,
            }
        
        loose_aggregator = MetricsAggregator(
            similarity_threshold=similarity_threshold,
            general_similarity_threshold=general_similarity_threshold,
            use_embedding=self.metrics_aggregator.use_embedding,
            embedding_api_config=embedding_config  # 复用配置（或从环境变量读取）
        )
        aggregated = loose_aggregator.deduplicate_metrics(collected)

        return aggregated

    def build_decision_prompt(
        self,
        aggregated_metrics: List[Dict[str, List[str]]],
        hard_cases: List[Dict],
        top_metrics: int = 12,
        top_hard_cases: int = 3,
        agents_catalog_path: Optional[str] = None,
        simple_format: bool = False
    ) -> str:
        """
        生成传给决策模型的prompt:
        - 汇总后的全局metrics(不区分sample)
        - 代表性难例简述
        - 期望决策模型输出: agent列表 + 每个agent的建议prompt
        
        参数:
            aggregated_metrics: 聚合后的指标列表
            hard_cases: 难例列表
            top_metrics: 最多显示的指标数量
            top_hard_cases: 最多显示的难例数量
            agents_catalog_path: agents_catalog.json 文件路径,如果提供则从中读取agents列表
            simple_format: 是否使用简易版格式,简易版只要求返回agents和prompts(一对多)
        """
        # 准备metrics文本
        metrics_lines = []
        for i, m in enumerate(aggregated_metrics[:top_metrics], 1):
            general = m.get("general", "")
            specifics = m.get("specific", [])
            specifics_str = "; ".join(specifics[:5])  # 控制长度
            metrics_lines.append(f"{i}. {general} -> {specifics_str}")

        # 准备难例摘要
        hard_lines = []
        for hc in hard_cases[:top_hard_cases]:
            sid = hc.get("sample_id", "unknown")
            diff = hc.get("difficulty_score", 0.0)
            conf = hc.get("confidence_score", 0.0)
            scores = []
            for name in ["weak", "normal", "sota"]:
                res = hc.get(name, {}) or {}
                c = res.get("correctness_score", None)
                t = res.get("training_quality_score", None)
                if c is not None and t is not None:
                    scores.append(f"{name}: c={c}, t={t}")
            scores_str = "; ".join(scores)
            hard_lines.append(
                f"- sample {sid}: difficulty={diff:.2f}, confidence={conf:.2f}; scores[{scores_str}]"
            )

        # 加载 agents 列表: 优先从文件读取,否则使用默认
        agent_requirements = []
        if agents_catalog_path:
            try:
                catalog_path = Path(agents_catalog_path)
                if catalog_path.is_file():
                    with catalog_path.open("r", encoding="utf-8") as f:
                        catalog = json.load(f)
                    agents = catalog.get("agents", [])
                    for agent in agents:
                        name = agent.get("name", "unknown")
                        specialty = agent.get("specialty", [])
                        specialty_str = ", ".join(specialty) if isinstance(specialty, list) else str(specialty)
                        models = agent.get("suggested_models", [])
                        models_str = ", ".join(models[:3]) if models else "N/A"
                        agent_requirements.append(
                            f"{name}: specialty=[{specialty_str}], suggested_models=[{models_str}]"
                        )
                    logger.info(f"从 {agents_catalog_path} 加载了 {len(agent_requirements)} 个 agents")
                else:
                    logger.warning(f"agents_catalog 文件不存在: {agents_catalog_path}, 使用默认列表")
            except Exception as e:
                logger.warning(f"读取 agents_catalog 失败: {e}, 使用默认列表")
        
        # 如果没有从文件加载到agents,使用默认列表
        if not agent_requirements:
            agent_requirements = [
                "data_filter_agent: 给出基于指标的保留/剔除建议，关注高频/general指标",
                "edge_case_agent: 结合难例，总结风险模式与需要人工复核的情形",
                "consistency_agent: 检查模型分歧点，对分歧大的样本给出额外检查标准",
                "prompt_designer: 为上述agents生成精炼的执行prompt"
            ]

        prompt = []
        
        if simple_format:
            # 简易版: 只要求返回 agents 和 prompts
            prompt.append("You are a decision model. Based on the aggregated data quality metrics and hard cases, select appropriate agents and generate prompts for them.")
            prompt.append("")
            prompt.append("Your task:")
            prompt.append("Select relevant agents from the suggested list and generate one or more prompts for each selected agent.")
            prompt.append("Each agent can have multiple prompts if needed (e.g., different scenarios or tasks).")
            prompt.append("")
            prompt.append("Aggregated metrics (cross-sample, deduplicated, top items):")
            prompt.extend(metrics_lines if metrics_lines else ["(no metrics found)"])
            prompt.append("")
            prompt.append(f"Representative hard cases (up to {top_hard_cases}):")
            prompt.extend(hard_lines if hard_lines else ["(no hard cases)"])
            prompt.append("")
            prompt.append("Suggested agent types (you may adjust/improve):")
            for a in agent_requirements:
                prompt.append(f"- {a}")
            prompt.append("")
            prompt.append("Output format (simplified):")
            prompt.append("""{
  "agents": [
    {
      "name": "agent_name",
      "prompts": [
        "prompt 1 for this agent",
        "prompt 2 for this agent (if multiple prompts needed)",
        ...
      ]
    },
    ...
  ]
}""")
            prompt.append("")
            prompt.append("Note: Only include agents that are relevant based on the metrics and hard cases. Each agent can have one or more prompts.")
        else:
            # 完整版: 包含所有信息
            prompt.append("You are a decision model. You will receive aggregated data quality metrics and hard cases from sampled data.")
            prompt.append("Your tasks:")
            prompt.append("1) Propose a set of agents (name + 1-2 line purpose).")
            prompt.append("2) For each agent, craft a concise prompt that leverages the metrics and hard-case patterns.")
            prompt.append("3) Summarize the key screening standards implied by the metrics (focus on general + top specifics).")
            prompt.append("4) Highlight risk patterns from hard cases and how agents should handle them.")
            prompt.append("")
            prompt.append("Aggregated metrics (cross-sample, deduplicated, top items):")
            prompt.extend(metrics_lines if metrics_lines else ["(no metrics found)"])
            prompt.append("")
            prompt.append(f"Representative hard cases (up to {top_hard_cases}):")
            prompt.extend(hard_lines if hard_lines else ["(no hard cases)"])
            prompt.append("")
            prompt.append("Suggested agent types (you may adjust/improve):")
            for a in agent_requirements:
                prompt.append(f"- {a}")
            prompt.append("")
            prompt.append("Output format suggestion (you may enrich):")
            prompt.append("""{
  "agents": [
    {"name": "...", "purpose": "...", "prompt": "..."},
    ...
  ],
  "screening_standards": [
    {"general": "...", "specific_examples": ["...", "..."]}
  ],
  "hard_case_guidance": [
    {"pattern": "...", "recommended_checks": ["..."], "who_handles": "<agent_name>"}
  ]
}""")

        return "\n".join(prompt)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def main():
    """
    命令行入口:
      输入: 评估结果文件 (.json 或 .jsonl), 每条记录包含 weak/normal/sota 三模型结果
      输出: 指定目录下的
        - samples_metrics.json: 每样本去重后的指标
        - hard_cases.json: 难例列表
        - global_metrics.json: 跨样本宽松去重后的指标
        - decision_prompt.txt: 传递给决策模型的 prompt
    """
    parser = argparse.ArgumentParser(description="Multi-model metrics processor")
    parser.add_argument("input_file", help="评估结果文件路径 (.json 或 .jsonl)")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="输出目录 (默认: outputs)",
    )
    parser.add_argument(
        "--include-hard-in-samples",
        action="store_true",
        help="若指定, 难例也会出现在 samples_metrics 中",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="样本内去重阈值 (默认 0.75)",
    )
    parser.add_argument(
        "--general-sim-threshold",
        type=float,
        default=0.80,
        help="样本内 General 相似度阈值 (默认 0.80)",
    )
    parser.add_argument(
        "--global-sim-threshold",
        type=float,
        default=0.65,
        help="跨样本去重阈值 (默认 0.65, 更宽松)",
    )
    parser.add_argument(
        "--global-general-sim-threshold",
        type=float,
        default=0.75,
        help="跨样本 General 相似度阈值 (默认 0.75, 更宽松)",
    )
    parser.add_argument(
        "--embedding-config",
        type=str,
        default=None,
        help="Embedding API 配置文件路径 (JSON), 或使用环境变量 EMBEDDING_API_KEY 等",
    )
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        default=False,
        help="是否使用 Embedding API (默认: False, 使用 TF-IDF)",
    )
    parser.add_argument(
        "--agents-catalog",
        type=str,
        default=None,
        help="agents_catalog.json 文件路径,如果提供则从中读取agents列表用于prompt生成",
    )
    parser.add_argument(
        "--simple-format",
        action="store_true",
        default=False,
        help="使用简易版输出格式,只要求返回agents和prompts(一对多)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 embedding 配置
    embedding_api_config = None
    if args.use_embedding:
        if args.embedding_config:
            with open(args.embedding_config, 'r', encoding='utf-8') as f:
                embedding_api_config = json.load(f)
        else:
            # 从环境变量读取
            import os
            api_key = os.getenv("EMBEDDING_API_KEY")
            if api_key:
                embedding_api_config = {
                    "api_type": os.getenv("EMBEDDING_API_TYPE", "openai"),
                    "api_key": api_key,
                    "base_url": os.getenv("EMBEDDING_BASE_URL"),
                    "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                    "timeout": int(os.getenv("EMBEDDING_TIMEOUT", "30")),
                    "max_retries": int(os.getenv("EMBEDDING_MAX_RETRIES", "2")),
                }
                # 移除 None 值
                embedding_api_config = {k: v for k, v in embedding_api_config.items() if v is not None}

    processor = MultiModelMetricsProcessor(
        difficulty_threshold=3.0,
        score_diff_threshold=2.5,
        similarity_threshold=args.similarity_threshold,
        general_similarity_threshold=args.general_sim_threshold,
        use_embedding=args.use_embedding,
        embedding_api_config=embedding_api_config,
    )

    samples_metrics, hard_cases = processor.process_file_per_sample(
        str(input_path),
        include_hard_cases_in_samples=args.include_hard_in_samples,
    )

    global_metrics = processor.aggregate_all_metrics_from_samples(
        samples_metrics,
        similarity_threshold=args.global_sim_threshold,
        general_similarity_threshold=args.global_general_sim_threshold,
    )

    prompt_text = processor.build_decision_prompt(
        aggregated_metrics=global_metrics,
        hard_cases=hard_cases,
        agents_catalog_path=args.agents_catalog,
        simple_format=args.simple_format,
    )

    (output_dir / "samples_metrics.json").write_text(
        json.dumps(samples_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hard_cases.json").write_text(
        json.dumps(hard_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "global_metrics.json").write_text(
        json.dumps(global_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "decision_prompt.txt").write_text(
        prompt_text,
        encoding="utf-8",
    )

    logger.info(
        "处理完成 | 样本=%d | 难例=%d | 全局指标=%d | 输出目录=%s",
        len(samples_metrics),
        len(hard_cases),
        len(global_metrics),
        output_dir,
    )

if __name__=="__main__":
    main()

# /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/d_extracted_sample_data/extra_evaluation.json

# python metrics_aggregater.py /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/d_extracted_sample_data/extra_evaluation.json \
#     --output-dir /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/e_aggregated_data \
#     --include-hard-in-samples \
#     --similarity-threshold 0.75 \
#     --general-sim-threshold 0.80 \
#     --global-sim-threshold 0.65 \
#     --global-general-sim-threshold 0.75 \
#     --use-embedding \
#     --embedding-config embedding_config.json \
#     --agents-catalog /home/zhuxuzhou/VQA_Auto/whole_pipeline/decider/agents_catalog.json \
#     --simple-format

"""
大规模图文数据聚类系统 - 内存优化版
支持百万级数据处理，通过流式读取、批量计算、特征缓存避免内存爆炸
"""

import json
import numpy as np
import h5py
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import gc
import base64
from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
import torch

# ==================== 关键修复：在导入transformers之前处理cv2问题 ====================
# transformers内部会尝试导入cv2，如果cv2导入失败（如缺少libGL.so.1），会导致transformers导入失败
# 我们需要在transformers导入之前就创建好假cv2模块

import os
import sys
import types

# 设置环境变量避免cv2加载OpenGL
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

# 尝试导入imageio作为WebP的备选方案
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# 尝试导入cv2，如果失败则创建一个完整的假cv2模块
HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
    print("[INFO] cv2导入成功")
except (ImportError, OSError, RuntimeError) as e:
    # 捕获所有可能的错误（ImportError, OSError, RuntimeError）
    HAS_CV2 = False
    error_msg = str(e)
    error_type = type(e).__name__
    
    # 检查是否是libGL相关错误
    is_libgl_error = 'libGL' in error_msg or 'libGL.so' in error_msg
    
    if is_libgl_error or error_type == 'OSError':
        print(f"[WARN] cv2导入失败（{error_type}）: {error_msg[:100]}")
        print(f"      创建假的cv2模块供transformers使用...")
    else:
        print(f"[WARN] cv2导入失败（{error_type}）: {error_msg[:100]}")
    
    # 创建一个完整的假cv2模块，包含transformers可能需要的所有属性
    fake_cv2 = types.ModuleType('cv2')
    
    # 添加版本信息
    fake_cv2.__version__ = '4.5.0'
    
    # 添加transformers可能使用的常量和函数
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.IMREAD_GRAYSCALE = 0
    fake_cv2.IMREAD_UNCHANGED = -1
    fake_cv2.COLOR_BGR2RGB = 4
    fake_cv2.COLOR_RGB2BGR = 4
    fake_cv2.COLOR_BGR2GRAY = 6
    fake_cv2.COLOR_GRAY2RGB = 8
    
    # 添加函数（虽然不会被调用，但避免AttributeError）
    def fake_imread(*args, **kwargs):
        raise NotImplementedError("cv2 is not available (fake module)")
    
    def fake_imwrite(*args, **kwargs):
        raise NotImplementedError("cv2 is not available (fake module)")
    
    def fake_cvtColor(*args, **kwargs):
        raise NotImplementedError("cv2 is not available (fake module)")
    
    fake_cv2.imread = fake_imread
    fake_cv2.imwrite = fake_imwrite
    fake_cv2.cvtColor = fake_cvtColor
    
    # 将假模块注册到sys.modules，这样transformers导入时就会使用它
    sys.modules['cv2'] = fake_cv2
    
    if is_libgl_error:
        print(f"      [INFO] 已创建假的cv2模块，transformers可以继续导入")
        print(f"      [提示] 如需使用cv2功能，请安装: apt-get install libgl1-mesa-glx libglib2.0-0")

# 现在可以安全地导入transformers了
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构定义 ====================
@dataclass
class DataSample:
    """轻量级数据样本结构"""
    idx: int
    data_type: str
    cluster_id: int = -1
    # 注意：不在内存中保存完整content和feature，通过idx索引


# ==================== 1. 特征缓存管理 ====================
class FeatureCache:
    """特征缓存管理器 - 使用HDF5存储大规模特征"""
    
    def __init__(self, cache_dir: str, feature_dim: int = 1546):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        self.cache_files = {}
    
    def create_cache(self, cache_name: str, n_samples: int):
        """创建特征缓存文件"""
        cache_path = self.cache_dir / f"{cache_name}_features.h5"
        
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset(
                'features',
                shape=(n_samples, self.feature_dim),
                dtype='float32',
                chunks=(min(1000, n_samples), self.feature_dim)
            )
            f.create_dataset('processed', shape=(n_samples,), dtype='bool')
        
        self.cache_files[cache_name] = cache_path
        print(f"  ✅ 创建缓存: {cache_path} ({n_samples} x {self.feature_dim})")
    
    def write_batch(self, cache_name: str, indices: np.ndarray, features: np.ndarray):
        """写入批量特征"""
        cache_path = self.cache_files[cache_name]
        
        with h5py.File(cache_path, 'r+') as f:
            f['features'][indices] = features
            f['processed'][indices] = True
    
    def read_all(self, cache_name: str) -> np.ndarray:
        """读取所有特征"""
        cache_path = self.cache_files[cache_name]
        
        with h5py.File(cache_path, 'r') as f:
            features = f['features'][:]
        
        return features
    
    def cleanup(self):
        """清理缓存文件"""
        for cache_path in self.cache_files.values():
            if cache_path.exists():
                cache_path.unlink()
        print(f"  🧹 清理缓存完成")



# ==================== 4. 特征提取模块（批量优化）====================
class FeatureExtractor(ABC):
    """特征提取器抽象基类"""
    
    @abstractmethod
    def extract_batch(self, samples: List[Dict], data_type: str) -> np.ndarray:
        """批量提取特征（直接处理原始数据）"""
        pass


# class SimpleTextFeatureExtractor(FeatureExtractor):
#     """简单文本特征提取器"""
    
#     def __init__(self, embedding_dim: int = 768):
#         self.embedding_dim = embedding_dim
    
#     def extract_batch(self, samples: List[Dict], data_type: str) -> np.ndarray:
#         """批量提取文本特征"""
#         texts = [self._get_text(sample, data_type) for sample in samples]
#         features = np.array([self._text_to_embedding(text) for text in texts])
#         return features
    
#     def _get_text(self, sample: Dict, data_type: str) -> str:
#         """提取文本内容（智能检测字段）"""
#         # 简化：查找所有字符串字段并拼接
#         text_parts = []
#         for key, value in sample.items():
#             if isinstance(value, str) and len(value) > 0:
#                 # 排除图像路径
#                 if not any(ext in value.lower() for ext in ['.jpg', '.png', '.jpeg']):
#                     text_parts.append(value)
#         return " ".join(text_parts)
    
#     def _text_to_embedding(self, text: str) -> np.ndarray:
#         """文本转嵌入（哈希模拟）"""
#         np.random.seed(hash(text) % (2**32))
#         embedding = np.random.randn(self.embedding_dim)
#         return normalize(embedding.reshape(1, -1))[0]


class VQAFeatureExtractor:
    """
    VQA数据多模态特征提取器
    
    特征提取策略:
    1. 图像特征: CLIP视觉编码器提取全局特征
    2. 问题特征: CLIP文本编码器 + 问题类型/长度等统计特征
    3. 答案特征: CLIP文本编码器 + 答案长度/类型等统计特征
    4. 交互特征: 问答对的语义相似度、多轮对话特征
    5. 多模态对齐特征: 图像-问题、图像-答案的跨模态相似度
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 normalize: bool = True,
                 feature_config: Dict[str, bool] = None):
        """
        Args:
            model_name: CLIP模型名称
            device: 运行设备
            normalize: 是否标准化特征
            feature_config: 特征选择配置字典，控制使用哪些特征
                可选配置项:
                - 'image': 图像特征 (512维)
                - 'question': 问题语义特征 (512维)
                - 'answer': 答案语义特征 (512维)
                - 'statistical': 统计特征 (10维)
                - 'interaction': 交互特征 (9维)
                默认: 使用所有特征
                示例: {'image': False, 'question': True, 'answer': False, 'statistical': False, 'interaction': False}
                     表示只使用问题特征进行聚类
        """
        self.device = device
        self.normalize = normalize
        
        # 特征配置（默认全部启用）
        default_config = {
            'image': True,
            'question': True,
            'answer': True,
            'statistical': True,
            'interaction': True
        }
        if feature_config is None:
            feature_config = default_config
        else:
            # 合并用户配置和默认配置
            for key in default_config:
                if key not in feature_config:
                    feature_config[key] = default_config[key]
        
        self.feature_config = feature_config
        
        # 加载CLIP模型
        print(f"🔄 加载CLIP模型: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # 特征标准化器
        self.scaler = StandardScaler() if normalize else None
        
        # 打印特征配置
        enabled_features = [k for k, v in self.feature_config.items() if v]
        disabled_features = [k for k, v in self.feature_config.items() if not v]
        print(f"✅ 模型加载完成 (设备: {device})")
        print(f"📊 特征配置:")
        print(f"   启用: {', '.join(enabled_features) if enabled_features else '无'}")
        if disabled_features:
            print(f"   禁用: {', '.join(disabled_features)}")
    
    def extract_batch(self, samples: List[Dict]) -> np.ndarray:
        """
        批量提取VQA样本的多模态特征
        
        Args:
            samples: VQA样本列表,每个样本包含:
                - dialogue: [{question: str, answer: str}, ...]
                - image_buffer_list: [{buffer: str, image_id: str}, ...]
                - task_type: str
                - source: str
        
        Returns:
            特征矩阵 (n_samples, feature_dim)
        """
        print(f"\n{'='*70}")
        print(f"🚀 开始批量特征提取: {len(samples)} 个VQA样本")
        print(f"{'='*70}\n")
        
        all_features = []
        
        for idx, sample in enumerate(samples):
            if (idx + 1) % 10 == 0:
                print(f"进度: {idx + 1}/{len(samples)}")
            
            features = self._extract_single_sample(sample, idx)
            all_features.append(features)
        
        # 堆叠所有特征
        feature_matrix = np.vstack(all_features)
        
        # 标准化
        if self.normalize and len(samples) > 1:
            print(f"\n🔧 标准化特征...")
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        print(f"\n{'='*70}")
        print(f"✅ 特征提取完成!")
        print(f"   特征维度: {feature_matrix.shape}")
        print(f"   特征范围: [{feature_matrix.min():.3f}, {feature_matrix.max():.3f}]")
        print(f"{'='*70}\n")
        
        return feature_matrix
    
    def _extract_single_sample(self, sample: Dict, idx: int) -> np.ndarray:
        """
        提取单个样本的综合特征（根据feature_config选择性组合）
        """
        feature_parts = []
        
        # 1. 提取图像特征（如果启用，或者交互特征需要）
        image_features = None
        if self.feature_config.get('image', True) or self.feature_config.get('interaction', True):
            image_features = self._extract_image_features(sample, idx)
            if self.feature_config.get('image', True):
                feature_parts.append(image_features)
        
        # 2. 提取对话特征（细化后的版本）
        dialogue_features_dict = self._extract_dialogue_features(sample, idx)
        
        # 2a. 问题特征（如果启用）
        if self.feature_config.get('question', True) and 'question' in dialogue_features_dict:
            feature_parts.append(dialogue_features_dict['question'])
        
        # 2b. 答案特征（如果启用）
        if self.feature_config.get('answer', True) and 'answer' in dialogue_features_dict:
            feature_parts.append(dialogue_features_dict['answer'])
        
        # 3. 提取统计特征（如果启用）
        if self.feature_config.get('statistical', True):
            stat_features = self._extract_statistical_features(sample)
            feature_parts.append(stat_features)
        
        # 4. 提取多模态交互特征（如果启用）
        if self.feature_config.get('interaction', True):
            # 需要图像和对话特征来计算交互特征
            # 如果图像特征未启用，仍然需要计算（但不加入最终特征）
            if image_features is None:
                image_features = self._extract_image_features(sample, idx)
            
            # 组合对话特征（用于交互特征计算）
            dialogue_features_combined = np.concatenate([
                dialogue_features_dict.get('question', np.zeros(512)),
                dialogue_features_dict.get('answer', np.zeros(512))
            ])
            
            interaction_features = self._extract_interaction_features(
                sample, image_features, dialogue_features_combined
            )
            feature_parts.append(interaction_features)
        
        # 5. 合并所有启用的特征
        if not feature_parts:
            raise ValueError("至少需要启用一个特征类型！请检查feature_config配置。")
        
        combined = np.concatenate(feature_parts)
        return combined
    
    def _extract_image_features(self, sample: Dict, idx: int) -> np.ndarray:
        """提取图像特征"""
        images = self._load_images(sample)
        
        if not images:
            # 没有图像时返回零向量
            print(f"[WARN] 样本 {idx} 未加载到任何图像，返回零向量。")
            return np.zeros(512)
        
        with torch.no_grad():
            # 处理多张图片: 取平均
            image_embeddings = []
            for img in images:
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = self.model.get_image_features(**inputs)
                image_embeddings.append(embedding.cpu().numpy())
            
            # 平均池化
            avg_embedding = np.mean(image_embeddings, axis=0).flatten()
        
        return avg_embedding
    
    def _extract_dialogue_features(self, sample: Dict, idx: int) -> Dict[str, np.ndarray]:
        """
        提取对话特征 (问题+答案的语义特征)
        
        Returns:
            字典，包含:
            - 'question': 问题语义特征 (512维)
            - 'answer': 答案语义特征 (512维)
        """
        dialogue = sample.get('dialogue', [])
        
        result = {}
        
        if not dialogue:
            if self.feature_config.get('question', True):
                result['question'] = np.zeros(512)
            if self.feature_config.get('answer', True):
                result['answer'] = np.zeros(512)
            return result
        
        with torch.no_grad():
            # 提取问题特征（如果启用）
            if self.feature_config.get('question', True):
                questions = [qa.get('question', '') for qa in dialogue]
                question_embeddings = self._encode_texts(questions)
                # 对多轮对话取平均
                result['question'] = np.mean(question_embeddings, axis=0)
            
            # 提取答案特征（如果启用）
            if self.feature_config.get('answer', True):
                answers = [qa.get('answer', '') for qa in dialogue]
                answer_embeddings = self._encode_texts(answers)
                # 对多轮对话取平均
                result['answer'] = np.mean(answer_embeddings, axis=0)
        
        return result
    
    def _extract_statistical_features(self, sample: Dict) -> np.ndarray:
        """提取统计特征"""
        dialogue = sample.get('dialogue', [])
        
        features = []
        
        # 1. 对话轮数
        features.append(len(dialogue))
        
        # 2. 问题统计
        if dialogue:
            questions = [qa.get('question', '') for qa in dialogue]
            features.extend([
                np.mean([len(q.split()) for q in questions]),  # 平均问题长度
                np.std([len(q.split()) for q in questions]),   # 问题长度标准差
                np.mean([len(q) for q in questions]),          # 平均字符数
            ])
        else:
            features.extend([0, 0, 0])
        
        # 3. 答案统计
        if dialogue:
            answers = [qa.get('answer', '') for qa in dialogue]
            features.extend([
                np.mean([len(a.split()) for a in answers]),    # 平均答案长度
                np.std([len(a.split()) for a in answers]),     # 答案长度标准差
                np.mean([len(a) for a in answers]),            # 平均字符数
            ])
        else:
            features.extend([0, 0, 0])
        
        # 4. 图像数量
        image_list = sample.get('image_buffer_list', [])
        features.append(len(image_list))
        
        # 5. 任务类型编码 (one-hot简化版)
        task_type = sample.get('task_type', '')
        task_indicators = [
            int('vqa' in task_type.lower()),
            int('caption' in task_type.lower()),
            int('turn' in task_type.lower()),
        ]
        features.extend(task_indicators)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_interaction_features(self, sample: Dict, 
                                     image_features: np.ndarray,
                                     dialogue_features: np.ndarray) -> np.ndarray:
        """提取多模态交互特征"""
        features = []
        
        dialogue = sample.get('dialogue', [])
        images = self._load_images(sample)
        has_image = len(images) > 0
        
        if not dialogue or not has_image:
            # 与特征命名保持一致的 9 维占位
            if not dialogue:
                print("[WARN] 对话为空，交互特征返回零向量。")
            if not has_image:
                print("[WARN] 图像未加载成功，交互特征返回零向量。")
            return np.zeros(9, dtype=np.float32)
        
        # 分离问题和答案特征
        question_features = dialogue_features[:512]
        answer_features = dialogue_features[512:]
        
        # 1. 图像-问题相似度
        img_q_sim = self._cosine_similarity(image_features, question_features)
        features.append(img_q_sim)
        
        # 2. 图像-答案相似度
        img_a_sim = self._cosine_similarity(image_features, answer_features)
        features.append(img_a_sim)
        
        # 3. 问题-答案相似度
        q_a_sim = self._cosine_similarity(question_features, answer_features)
        features.append(q_a_sim)
        
        # 4. 问题中是否包含图像引用标记
        questions = [qa.get('question', '') for qa in dialogue]
        ##### here111
        # 兼容旧格式 "<image>" 和新格式 "[Image_{idx}]"
        has_image_ref = any(('<image>' in q) or ('[Image_' in q) for q in questions)
        ##### here222
        features.append(float(has_image_ref))
        
        # 5. 多轮对话连贯性 (相邻QA对的相似度)
        if len(dialogue) > 1:
            coherence_scores = []
            for i in range(len(dialogue) - 1):
                curr_text = dialogue[i]['question'] + ' ' + dialogue[i]['answer']
                next_text = dialogue[i+1]['question'] + ' ' + dialogue[i+1]['answer']
                
                with torch.no_grad():
                    curr_emb = self._encode_texts([curr_text])[0]
                    next_emb = self._encode_texts([next_text])[0]
                    sim = self._cosine_similarity(curr_emb, next_emb)
                    coherence_scores.append(sim)
            
            features.append(np.mean(coherence_scores))
            features.append(np.std(coherence_scores))
        else:
            features.extend([0, 0])
        
        # 6. 问题复杂度指标
        avg_question_words = np.mean([len(qa['question'].split()) for qa in dialogue])
        features.append(avg_question_words / 50)  # 归一化
        
        # 7. 答案详细度指标
        avg_answer_words = np.mean([len(qa['answer'].split()) for qa in dialogue])
        features.append(avg_answer_words / 100)  # 归一化
        
        # 8. 问答比例
        if avg_question_words > 0:
            features.append(avg_answer_words / avg_question_words)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        if not texts:
            return np.zeros((1, 512))
        
        inputs = self.processor(text=texts, return_tensors="pt", 
                               padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        
        return embeddings.cpu().numpy()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _load_images(self, sample: Dict) -> List[Image.Image]:
        """加载样本中的所有图像"""
        images = []
        image_buffer_list = sample.get('image_buffer_list', [])
        
        for i, img_data in enumerate(image_buffer_list):
            buffer = img_data.get('buffer', '')
            if not buffer:
                print(f"[WARN] 样本中第 {i} 个 image_buffer 为空，跳过。")
                continue
            
            img = self._base64_to_image(buffer)
            if img is None:
                # 打印出错信息（避免打印整个 base64，只给前缀）
                buf_preview = str(buffer)
                if len(buf_preview) > 60:
                    buf_preview = buf_preview[:60] + "...[truncated]"
                print(f"[ERROR] 无法从第 {i} 个 image_buffer 解码图像，buffer 前缀: {buf_preview}")
                continue
            
            images.append(img)
        
        return images
    
    def _base64_to_image(self, img_b64: Any) -> Optional[Image.Image]:
        """Base64转PIL Image，兼容 bytes / b"..." 字符串等多种格式，支持WebP格式"""
        img_bytes = None
        original_input_type = type(img_b64).__name__
        original_input_length = len(str(img_b64)) if img_b64 else 0
        data_source = "unknown"  # 记录数据来源
        
        try:
            # ========== 步骤1: 检测输入是否已经是二进制数据 ==========
            # 如果输入是bytes类型，先检查是否是图片二进制数据（不是base64编码的字符串）
            if isinstance(img_b64, bytes):
                # 检查是否是图片文件头（RIFF/WEBP, JPEG, PNG等）
                if len(img_b64) >= 12:
                    if (img_b64[:4] == b'RIFF' and img_b64[8:12] == b'WEBP') or \
                       img_b64[:2] == b'\xff\xd8' or \
                       img_b64[:8] == b'\x89PNG\r\n\x1a\n':
                        # 已经是二进制图片数据，直接使用
                        print(f"[DEBUG] 检测到输入是二进制图片数据（{original_input_type}），跳过Base64解码")
                        img_bytes = img_b64
                        data_source = "direct_binary"
                    else:
                        # 可能是base64编码的bytes，尝试解码为字符串
                        try:
                            img_b64 = img_b64.decode("utf-8")
                        except Exception:
                            img_b64 = img_b64.decode("latin-1", errors="ignore")
                else:
                    # 太短，尝试解码为字符串
                    try:
                        img_b64 = img_b64.decode("utf-8")
                    except Exception:
                        img_b64 = img_b64.decode("latin-1", errors="ignore")
            else:
                img_b64 = str(img_b64)

            # ========== 步骤2: 如果已经是二进制数据，跳过Base64解码 ==========
            if img_bytes is not None:
                pass  # 已经设置好，继续后续处理
            else:
                # ========== 步骤2a: 处理字符串格式的输入 ==========
                # 检查是否是Python字符串表示的二进制数据（b'...' 或包含转义序列）
                is_python_bytes_literal = False
                if img_b64.startswith("b'") and img_b64.endswith("'"):
                    # 这是 b'...' 格式的字符串表示
                    try:
                        import ast
                        img_bytes = ast.literal_eval(img_b64)
                        data_source = "python_bytes_literal"
                        is_python_bytes_literal = True
                        print(f"[DEBUG] 检测到Python bytes字面量（b'...'），直接解析")
                    except Exception as eval_error:
                        print(f"[WARN] 解析b'...'格式失败: {eval_error}，尝试其他方法...")
                        img_b64 = img_b64[2:-1]  # 去掉 b' 和 '
                elif img_b64.startswith('b"') and img_b64.endswith('"'):
                    # 这是 b"..." 格式的字符串表示
                    try:
                        import ast
                        img_bytes = ast.literal_eval(img_b64)
                        data_source = "python_bytes_literal"
                        is_python_bytes_literal = True
                        print(f"[DEBUG] 检测到Python bytes字面量（b\"...\"），直接解析")
                    except Exception as eval_error:
                        print(f"[WARN] 解析b\"...\"格式失败: {eval_error}，尝试其他方法...")
                        img_b64 = img_b64[2:-1]  # 去掉 b" 和 "
                
                if not is_python_bytes_literal:
                    # 处理 data URI
                    if ',' in img_b64 and img_b64.startswith('data:'):
                        img_b64 = img_b64.split(',', 1)[1]
                    
                    # 清理并修复padding
                    img_b64_cleaned = img_b64.strip()
                    
                    # 检查是否是有效的base64字符
                    base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                    is_likely_base64 = len(img_b64_cleaned) > 0 and all(c in base64_chars or c.isspace() for c in img_b64_cleaned[:100])
                    
                    if not is_likely_base64:
                        print(f"[WARN] 输入可能不是有效的Base64字符串:")
                        print(f"  - 输入类型: {original_input_type}")
                        print(f"  - 输入长度: {original_input_length}")
                        print(f"  - 清理后长度: {len(img_b64_cleaned)}")
                        print(f"  - 前100字符: {img_b64_cleaned[:100]}")
                        
                        # 检查是否是Python字符串表示的二进制数据（包含\x转义序列）
                        has_escape_sequences = '\\x' in img_b64_cleaned or '\\n' in img_b64_cleaned or '\\t' in img_b64_cleaned
                        
                        if has_escape_sequences:
                            print(f"  - 检测到转义序列，尝试解析为Python字符串表示的二进制数据...")
                            try:
                                # 使用ast.literal_eval安全地解析字符串
                                import ast
                                # 如果字符串以b'或b"开头，需要特殊处理
                                if img_b64_cleaned.startswith("b'") or img_b64_cleaned.startswith('b"'):
                                    # 已经是b'...'格式，直接解析
                                    img_bytes = ast.literal_eval(img_b64_cleaned)
                                    data_source = "python_string_literal"
                                    print(f"  - [成功] 解析Python字符串字面量，得到 {len(img_bytes)} bytes")
                                else:
                                    # 尝试添加b前缀
                                    try:
                                        img_bytes = ast.literal_eval('b"' + img_b64_cleaned.replace('"', '\\"') + '"')
                                        data_source = "python_string_literal"
                                        print(f"  - [成功] 解析Python字符串字面量（添加b前缀），得到 {len(img_bytes)} bytes")
                                    except:
                                        # 尝试使用codecs.decode处理转义序列
                                        import codecs
                                        img_bytes = codecs.decode(img_b64_cleaned, 'unicode_escape').encode('latin-1')
                                        data_source = "unicode_escape_decoded"
                                        print(f"  - [成功] 使用unicode_escape解码，得到 {len(img_bytes)} bytes")
                            except Exception as parse_error:
                                print(f"  - [失败] 解析失败: {type(parse_error).__name__}: {parse_error}")
                                # 最后尝试：直接使用latin-1编码
                                try:
                                    img_bytes = img_b64_cleaned.encode('latin-1')
                                    data_source = "latin1_encoded"
                                    print(f"  - [回退] 使用latin-1编码，得到 {len(img_bytes)} bytes")
                                except:
                                    pass
                        else:
                            # 没有转义序列，尝试直接编码
                            print(f"  - 尝试直接作为二进制数据处理...")
                            try:
                                img_bytes = img_b64_cleaned.encode('latin-1')
                                data_source = "latin1_encoded"
                            except:
                                pass
                
                if img_bytes is None:
                    # 尝试Base64解码
                    padding = len(img_b64_cleaned) % 4
                    if padding:
                        img_b64_cleaned += '=' * (4 - padding)
                    
                    try:
                        img_bytes = base64.b64decode(img_b64_cleaned, validate=True)
                        data_source = "base64_decoded"
                    except Exception as decode_error:
                        print(f"[ERROR] Base64解码失败:")
                        print(f"  - 输入类型: {original_input_type}")
                        print(f"  - 输入长度: {original_input_length}")
                        print(f"  - Base64字符串长度: {len(img_b64_cleaned)}")
                        print(f"  - 错误类型: {type(decode_error).__name__}")
                        print(f"  - 错误信息: {decode_error}")
                        print(f"  - Base64前缀(前200字符): {img_b64_cleaned[:200] if len(img_b64_cleaned) > 200 else img_b64_cleaned}")
                        return None
            
            # ========== 步骤3: 详细格式检测 ==========
            file_size = len(img_bytes)
            file_header = img_bytes[:16] if len(img_bytes) >= 16 else img_bytes
            header_hex = ' '.join(f'{b:02x}' for b in file_header[:12])
            header_ascii = ''.join(chr(b) if 32 <= b < 127 else '.' for b in file_header[:12])
            
            # 检测图片格式
            is_webp = False
            is_jpeg = False
            is_png = False
            webp_subformat = None
            
            print(f"[DEBUG] 图片数据检测:")
            print(f"  - 数据来源: {data_source if img_bytes is not None else 'Base64解码'}")
            print(f"  - 文件大小: {file_size} bytes")
            print(f"  - 文件头(hex): {header_hex}")
            print(f"  - 文件头(ascii): {header_ascii}")
            
            if file_size >= 12:
                if img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                    is_webp = True
                    # 检测WebP子格式
                    if file_size >= 16:
                        chunk_type = img_bytes[12:16]
                        if chunk_type == b'VP8 ':
                            webp_subformat = 'VP8 (lossy)'
                        elif chunk_type == b'VP8L':
                            webp_subformat = 'VP8L (lossless)'
                        elif chunk_type == b'VP8X':
                            webp_subformat = 'VP8X (extended)'
                        else:
                            webp_subformat = f'Unknown chunk: {chunk_type}'
                    print(f"  - 格式: WebP ({webp_subformat})")
                elif img_bytes[:2] == b'\xff\xd8':
                    is_jpeg = True
                    print(f"  - 格式: JPEG")
                elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    is_png = True
                    print(f"  - 格式: PNG")
                else:
                    print(f"  - 格式: 未知（前4字节: {img_bytes[:4]}）")
            else:
                print(f"  - 格式: 数据太短（{file_size} < 12 bytes）")
            
            # ========== 步骤4: 尝试用PIL打开 ==========
            try:
                img_buffer = BytesIO(img_bytes)
                img = Image.open(img_buffer)
                # 如果是RGBA或其他模式，转换为RGB
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                return img
            except (UnidentifiedImageError, OSError, IOError) as pil_error:
                # PIL无法识别，输出详细诊断信息
                print(f"[DEBUG] PIL无法识别图片:")
                print(f"  - 文件大小: {file_size} bytes")
                print(f"  - 文件头(hex): {header_hex}")
                print(f"  - 文件头(ascii): {header_ascii}")
                print(f"  - 格式检测: WebP={is_webp}, JPEG={is_jpeg}, PNG={is_png}")
                if is_webp:
                    print(f"  - WebP子格式: {webp_subformat}")
                print(f"  - PIL错误类型: {type(pil_error).__name__}")
                print(f"  - PIL错误信息: {pil_error}")
                
                # 尝试用备选方案（特别是WebP格式）
                if is_webp:
                    print(f"  - 尝试备选方案加载WebP...")
                    
                    # 方案1: 尝试使用imageio
                    if HAS_IMAGEIO:
                        try:
                            print(f"    [尝试1] 使用imageio加载...")
                            img_array = imageio.imread(img_bytes, format='webp')
                            img = Image.fromarray(img_array)
                            if img.mode != 'RGB':
                                img = img.convert("RGB")
                            print(f"    [成功] imageio成功加载WebP")
                            return img
                        except Exception as imageio_error:
                            print(f"    [失败] imageio加载失败: {type(imageio_error).__name__}: {imageio_error}")
                    
                    # 方案2: 尝试使用cv2 (OpenCV)
                    if HAS_CV2:
                        try:
                            print(f"    [尝试2] 使用cv2加载...")
                            # cv2.imdecode需要numpy数组
                            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img_array is not None:
                                # cv2使用BGR格式，需要转换为RGB
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(img_array)
                                print(f"    [成功] cv2成功加载WebP")
                                return img
                            else:
                                print(f"    [失败] cv2.imdecode返回None")
                        except Exception as cv2_error:
                            print(f"    [失败] cv2加载失败: {type(cv2_error).__name__}: {cv2_error}")
                    
                    # 如果所有方案都失败
                    print(f"  [总结] 所有WebP加载方案均失败")
                    if not HAS_IMAGEIO and not HAS_CV2:
                        print(f"      建议安装: pip install imageio imageio-ffmpeg 或 pip install opencv-python-headless")
                    raise pil_error
                else:
                    # 不是WebP格式，直接抛出原始错误
                    print(f"  [总结] 未知图片格式，无法加载")
                    raise pil_error
            
        except Exception as e:
            # 最终错误处理，输出所有诊断信息
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"[ERROR] _base64_to_image 最终失败:")
            print(f"  - 错误类型: {error_type}")
            print(f"  - 错误信息: {error_msg}")
            print(f"  - 输入类型: {original_input_type}")
            print(f"  - 输入长度: {original_input_length}")
            
            if img_bytes is not None:
                print(f"  - 解码后大小: {len(img_bytes)} bytes")
                print(f"  - 文件头(hex): {' '.join(f'{b:02x}' for b in img_bytes[:12])}")
                print(f"  - 文件头(ascii): {''.join(chr(b) if 32 <= b < 127 else '.' for b in img_bytes[:12])}")
                
                # 再次检测格式
                if len(img_bytes) >= 12:
                    if img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                        print(f"  - 格式: WebP")
                        if len(img_bytes) >= 16:
                            chunk_type = img_bytes[12:16]
                            print(f"  - WebP chunk: {chunk_type}")
                    elif img_bytes[:2] == b'\xff\xd8':
                        print(f"  - 格式: JPEG")
                    elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                        print(f"  - 格式: PNG")
                    else:
                        print(f"  - 格式: 未知")
            else:
                print(f"  - Base64解码失败，无法获取图片数据")
            
            return None
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称(用于分析)，根据feature_config返回实际使用的特征名称
        """
        names = []
        
        # 图像特征
        if self.feature_config.get('image', True):
            names.extend([f"image_feat_{i}" for i in range(512)])
        
        # 对话特征
        if self.feature_config.get('question', True):
            names.extend([f"question_feat_{i}" for i in range(512)])
        
        if self.feature_config.get('answer', True):
            names.extend([f"answer_feat_{i}" for i in range(512)])
        
        # 统计特征
        if self.feature_config.get('statistical', True):
            names.extend([
                'dialogue_turns',
                'avg_question_words', 'std_question_words', 'avg_question_chars',
                'avg_answer_words', 'std_answer_words', 'avg_answer_chars',
                'num_images',
                'is_vqa', 'is_caption', 'is_multiturn'
            ])
        
        # 交互特征
        if self.feature_config.get('interaction', True):
            names.extend([
                'img_question_sim', 'img_answer_sim', 'question_answer_sim',
                'has_image_reference', 'dialogue_coherence_mean', 'dialogue_coherence_std',
                'question_complexity', 'answer_detail', 'answer_question_ratio'
            ])
        
        return names


# ==================== 2. 流式数据加载器 ====================
class VQADataLoader:
    """VQA数据流式加载器"""
    
    def __init__(self, data_path: str, batch_size: int = 100):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
    
    def stream_batches(self):
        """流式读取JSONL数据"""
        batch = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    batch.append(sample)
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
        
        # 返回最后一批
        if batch:
            yield batch
    
    def count_samples(self) -> int:
        """统计样本总数"""
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


# ==================== 3. 聚类算法 ====================
class ClusteringAlgorithm(ABC):
    """聚类算法抽象基类"""
    
    @abstractmethod
    def fit_predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """执行聚类"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取算法名称"""
        pass


class AutoMiniBatchKMeans(ClusteringAlgorithm):
    """自动选择最优簇数的Mini-Batch KMeans"""
    
    def __init__(self, random_state: int = 42, batch_size: int = 1000):
        self.random_state = random_state
        self.batch_size = batch_size
        self.model = None
        self.best_k = None
        self.scores = {}
    
    def fit_predict(self, features: np.ndarray, 
                    n_clusters: Optional[int] = None,
                    min_clusters: int = 5, 
                    max_clusters: int = 20,
                    auto_select: bool = True) -> np.ndarray:
        """
        执行聚类,支持自动选择簇数
        
        Args:
            features: 特征矩阵
            n_clusters: 指定簇数(如果为None则自动选择)
            min_clusters: 最小簇数
            max_clusters: 最大簇数
            auto_select: 是否自动选择最优簇数
        """
        
        # 如果指定了簇数,直接聚类
        if n_clusters is not None:
            return self._cluster_with_k(features, n_clusters)
        
        # 自动选择最优簇数
        if not auto_select:
            n_clusters = min_clusters
            return self._cluster_with_k(features, n_clusters)
        
        print(f"\n  🔍 自动选择最优簇数 (范围: {min_clusters}-{max_clusters})...")
        
        # 采样用于快速评估
        n_samples = len(features)
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features[sample_indices]
        
        # 尝试不同的k值
        max_clusters = min(max_clusters, sample_size // 2)
        best_score = -1
        best_k = min_clusters
        
        for k in range(min_clusters, max_clusters + 1):
            try:
                # 临时模型
                model = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    batch_size=min(self.batch_size, sample_size),
                    max_iter=100
                )
                labels = model.fit_predict(sample_features)
                
                # 检查是否产生了有效的簇
                unique_labels = len(np.unique(labels))
                if unique_labels <= 1:
                    continue
                
                # 计算轮廓系数
                score = silhouette_score(
                    sample_features, 
                    labels, 
                    sample_size=min(5000, sample_size)
                )
                
                # 同时考虑Calinski-Harabasz指数(奖励紧密且分离的簇)
                ch_score = calinski_harabasz_score(sample_features, labels)
                
                # 组合得分(可调整权重)
                combined_score = 0.7 * score + 0.3 * (ch_score / 10000)
                
                self.scores[k] = {
                    'silhouette': score,
                    'calinski_harabasz': ch_score,
                    'combined': combined_score
                }
                
                print(f"    k={k}: silhouette={score:.3f}, CH={ch_score:.1f}, combined={combined_score:.3f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    
            except Exception as e:
                print(f"    k={k}: 评估失败 ({e})")
                continue
        
        self.best_k = best_k
        print(f"  ✅ 选择最优簇数: k={best_k} (得分: {best_score:.3f})")
        
        # 使用最优k值进行最终聚类
        return self._cluster_with_k(features, best_k)
    
    def _cluster_with_k(self, features: np.ndarray, k: int) -> np.ndarray:
        """使用指定k值聚类"""
        print(f"  🎯 使用 k={k} 进行聚类...")
        
        self.model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.random_state,
            batch_size=self.batch_size,
            max_iter=300,
            n_init=10
        )
        
        labels = self.model.fit_predict(features)
        
        # 统计簇大小
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  📊 簇分布:")
        for cluster_id, count in zip(unique, counts):
            print(f"    簇 {cluster_id}: {count} 个样本 ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def get_name(self) -> str:
        return "MiniBatchKMeans"


class HierarchicalClustering(ClusteringAlgorithm):
    """层次聚类(适合中小规模数据)，支持自动选择最优簇数"""
    
    def __init__(self, linkage: str = 'ward', distance_threshold: float = None):
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.model = None
        self.best_k = None
        self.scores = {}
    
    def fit_predict(self, features: np.ndarray, 
                    n_clusters: Optional[int] = None,
                    min_clusters: int = 5,
                    max_clusters: int = 20,
                    auto_select: bool = True,
                    **kwargs) -> np.ndarray:
        """
        执行层次聚类，支持自动选择最优簇数
        
        Args:
            features: 特征矩阵
            n_clusters: 指定簇数(如果为None则自动选择)
            min_clusters: 最小簇数
            max_clusters: 最大簇数
            auto_select: 是否自动选择最优簇数
        """
        
        if len(features) > 10000:
            print(f"  ⚠️  数据量较大({len(features)}),层次聚类可能较慢")
        
        # 如果指定了簇数,直接聚类
        if n_clusters is not None:
            return self._cluster_with_k(features, n_clusters)
        
        # 如果使用distance_threshold，直接聚类
        if self.distance_threshold is not None:
            return self._cluster_with_threshold(features)
        
        # 自动选择最优簇数
        if not auto_select:
            n_clusters = min_clusters
            return self._cluster_with_k(features, n_clusters)
        
        print(f"\n  🔍 自动选择最优簇数 (范围: {min_clusters}-{max_clusters})...")
        
        # 采样用于快速评估（层次聚类计算量大）
        n_samples = len(features)
        sample_size = min(5000, n_samples)  # 层次聚类采样更小
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features[sample_indices]
        
        # 尝试不同的k值
        max_clusters = min(max_clusters, sample_size // 2)
        best_score = -1
        best_k = min_clusters
        
        for k in range(min_clusters, max_clusters + 1):
            try:
                # 临时模型
                model = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=self.linkage
                )
                labels = model.fit_predict(sample_features)
                
                # 检查是否产生了有效的簇
                unique_labels = len(np.unique(labels))
                if unique_labels <= 1:
                    continue
                
                # 计算轮廓系数
                score = silhouette_score(
                    sample_features, 
                    labels, 
                    sample_size=min(3000, sample_size)
                )
                
                # 同时考虑Calinski-Harabasz指数
                ch_score = calinski_harabasz_score(sample_features, labels)
                
                # 组合得分(可调整权重)
                combined_score = 0.7 * score + 0.3 * (ch_score / 10000)
                
                self.scores[k] = {
                    'silhouette': score,
                    'calinski_harabasz': ch_score,
                    'combined': combined_score
                }
                
                print(f"    k={k}: silhouette={score:.3f}, CH={ch_score:.1f}, combined={combined_score:.3f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    
            except Exception as e:
                print(f"    k={k}: 评估失败 ({e})")
                continue
        
        self.best_k = best_k
        print(f"  ✅ 选择最优簇数: k={best_k} (得分: {best_score:.3f})")
        
        # 使用最优k值进行最终聚类
        return self._cluster_with_k(features, best_k)
    
    def _cluster_with_k(self, features: np.ndarray, k: int) -> np.ndarray:
        """使用指定k值聚类"""
        print(f"  🎯 使用 k={k} 进行层次聚类...")
        
        self.model = AgglomerativeClustering(
            n_clusters=k,
            linkage=self.linkage
        )
        
        labels = self.model.fit_predict(features)
        
        # 统计簇大小
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  📊 簇分布:")
        for cluster_id, count in zip(unique, counts):
            print(f"    簇 {cluster_id}: {count} 个样本 ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def _cluster_with_threshold(self, features: np.ndarray) -> np.ndarray:
        """使用distance_threshold聚类"""
        print(f"  🎯 使用 distance_threshold={self.distance_threshold} 进行层次聚类...")
        
        self.model = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            n_clusters=None
        )
        
        labels = self.model.fit_predict(features)
        
        n_clusters = len(np.unique(labels))
        print(f"  📊 生成了 {n_clusters} 个簇")
        
        return labels
    
    def get_name(self) -> str:
        return "Hierarchical"


class DensityClustering(ClusteringAlgorithm):
    """基于密度的聚类(DBSCAN)，支持自动选择最优参数"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.best_eps = None
        self.best_min_samples = None
        self.scores = {}
    
    def fit_predict(self, features: np.ndarray,
                    eps: Optional[float] = None,
                    min_samples: Optional[int] = None,
                    eps_range: Tuple[float, float] = (0.1, 2.0),
                    eps_steps: int = 10,
                    min_samples_range: Tuple[int, int] = (3, 10),
                    auto_select: bool = True,
                    **kwargs) -> np.ndarray:
        """
        执行DBSCAN聚类，支持自动选择最优参数
        
        Args:
            features: 特征矩阵
            eps: 指定eps参数(如果为None则自动选择)
            min_samples: 指定min_samples参数(如果为None则自动选择)
            eps_range: eps搜索范围 (min, max)
            eps_steps: eps搜索步数
            min_samples_range: min_samples搜索范围 (min, max)
            auto_select: 是否自动选择最优参数
        """
        
        # 如果指定了参数,直接聚类
        if eps is not None and min_samples is not None:
            self.eps = eps
            self.min_samples = min_samples
            return self._cluster_with_params(features)
        
        # 自动选择最优参数
        if not auto_select:
            if eps is not None:
                self.eps = eps
            if min_samples is not None:
                self.min_samples = min_samples
            return self._cluster_with_params(features)
        
        print(f"\n  🔍 自动选择最优DBSCAN参数...")
        
        # 采样用于快速评估
        n_samples = len(features)
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features[sample_indices]
        
        # 智能估计eps范围：使用k-距离图方法
        print(f"    📊 分析数据分布以估计eps范围...")
        
        # 方法1: 计算最近邻距离的统计信息
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(min_samples_range[1] + 1, sample_size - 1)
        if n_neighbors < 2:
            n_neighbors = 2
        
        try:
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            neighbors_fit = neighbors.fit(sample_features)
            distances, indices = neighbors_fit.kneighbors(sample_features)
            
            # 获取第k个最近邻的距离（k=min_samples）
            k_distances = distances[:, min_samples_range[0]:min_samples_range[1]+1].mean(axis=1)
            k_distances_sorted = np.sort(k_distances)
            
            # 使用分位数来估计eps范围
            # 通常eps应该在第50-90百分位之间
            eps_min_estimate = np.percentile(k_distances_sorted, 25)
            eps_max_estimate = np.percentile(k_distances_sorted, 90)
            
            # 如果估计的范围太小，使用特征标准差作为参考
            feature_std = np.std(sample_features)
            if eps_max_estimate < feature_std * 0.1:
                eps_max_estimate = feature_std * 0.5
            
            # 调整eps范围
            eps_min = max(eps_range[0], eps_min_estimate * 0.5)
            eps_max = min(eps_range[1], eps_max_estimate * 2.0)
            
            # 确保范围合理
            if eps_max <= eps_min:
                eps_max = eps_min * 3.0
            
            eps_range = (eps_min, eps_max)
            print(f"    ✅ 根据k-距离图估计eps范围: {eps_range[0]:.3f} - {eps_range[1]:.3f}")
            print(f"       (原始范围: {eps_range[0]:.3f} - {eps_range[1]:.3f})")
            print(f"       k-距离统计: min={k_distances_sorted[0]:.3f}, "
                  f"median={np.median(k_distances_sorted):.3f}, "
                  f"max={k_distances_sorted[-1]:.3f}")
        except Exception as e:
            print(f"    ⚠️  k-距离图分析失败: {e}，使用原始范围")
            # 如果k-距离图失败，使用特征标准差
            feature_std = np.std(sample_features)
            if eps_range[1] > feature_std * 3:
                eps_range = (eps_range[0], min(eps_range[1], feature_std * 2.0))
                print(f"    [调整] 根据特征标准差调整eps范围: {eps_range[0]:.3f} - {eps_range[1]:.3f}")
        
        print(f"    eps范围: {eps_range[0]:.3f} - {eps_range[1]:.3f} (步数: {eps_steps})")
        print(f"    min_samples范围: {min_samples_range[0]} - {min_samples_range[1]}")
        
        # 生成参数网格
        eps_values = np.linspace(eps_range[0], eps_range[1], eps_steps)
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
        
        best_score = -1
        best_eps = eps_values[0]
        best_min_samples = min_samples_values[0]
        
        total_combinations = len(eps_values) * len(min_samples_values)
        current_combination = 0
        
        for eps_val in eps_values:
            for min_samples_val in min_samples_values:
                current_combination += 1
                try:
                    # 临时模型
                    model = DBSCAN(eps=eps_val, min_samples=min_samples_val, n_jobs=-1)
                    labels = model.fit_predict(sample_features)
                    
                    # 统计结果
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels[unique_labels != -1])  # 排除噪声点
                    n_noise = (labels == -1).sum()
                    
                    # 检查是否产生了有效的簇
                    if n_clusters <= 1:
                        continue
                    
                    # 如果噪声点太多，跳过
                    noise_ratio = n_noise / len(labels)
                    if noise_ratio > 0.7:  # 噪声点超过70%，跳过（放宽阈值）
                        continue
                    
                    # 如果簇太少，也跳过（可能是eps太大）
                    if n_clusters == 0:
                        continue
                    
                    # 计算轮廓系数（只对非噪声点）
                    non_noise_mask = labels != -1
                    if non_noise_mask.sum() < 2:
                        continue
                    
                    try:
                        score = silhouette_score(
                            sample_features[non_noise_mask],
                            labels[non_noise_mask],
                            sample_size=min(5000, non_noise_mask.sum())
                        )
                    except:
                        continue
                    
                    # 同时考虑簇数和噪声比例
                    # 奖励：更多簇、更少噪声、更高轮廓系数
                    cluster_bonus = min(n_clusters / 20.0, 1.0)  # 簇数奖励（最多20个簇）
                    noise_penalty = noise_ratio  # 噪声惩罚
                    
                    # 组合得分（调整权重，更重视轮廓系数和簇数）
                    # 如果噪声点太多，大幅惩罚
                    if noise_ratio > 0.3:
                        combined_score = 0.4 * score + 0.2 * cluster_bonus - 0.4 * noise_penalty
                    else:
                        combined_score = 0.6 * score + 0.3 * cluster_bonus - 0.1 * noise_penalty
                    
                    param_key = f"eps={eps_val:.3f},min_samples={min_samples_val}"
                    self.scores[param_key] = {
                        'silhouette': score,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio,
                        'combined': combined_score
                    }
                    
                    if current_combination % 5 == 0 or combined_score > best_score:
                        print(f"    [{current_combination}/{total_combinations}] "
                              f"eps={eps_val:.3f}, min_samples={min_samples_val}: "
                              f"clusters={n_clusters}, noise={noise_ratio:.2%}, "
                              f"score={score:.3f}, combined={combined_score:.3f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_eps = eps_val
                        best_min_samples = min_samples_val
                        
                except Exception as e:
                    if current_combination % 10 == 0:
                        print(f"    [{current_combination}/{total_combinations}] "
                              f"eps={eps_val:.3f}, min_samples={min_samples_val}: 评估失败 ({e})")
                    continue
        
        self.best_eps = best_eps
        self.best_min_samples = best_min_samples
        print(f"  ✅ 选择最优参数: eps={best_eps:.3f}, min_samples={best_min_samples} (得分: {best_score:.3f})")
        
        # 使用最优参数进行最终聚类
        self.eps = best_eps
        self.min_samples = best_min_samples
        return self._cluster_with_params(features)
    
    def _cluster_with_params(self, features: np.ndarray) -> np.ndarray:
        """使用指定参数聚类"""
        print(f"  🎯 使用 eps={self.eps:.3f}, min_samples={self.min_samples} 进行DBSCAN聚类...")
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = self.model.fit_predict(features)
        
        # 统计噪声点和簇
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = (labels == -1).sum()
        
        print(f"  📊 生成了 {n_clusters} 个簇, {n_noise} 个噪声点 ({n_noise/len(labels)*100:.1f}%)")
        
        # 统计簇大小
        if n_clusters > 0:
            non_noise_labels = labels[labels != -1]
            if len(non_noise_labels) > 0:
                unique, counts = np.unique(non_noise_labels, return_counts=True)
                print(f"  📊 簇分布:")
                for cluster_id, count in zip(unique, counts):
                    print(f"    簇 {cluster_id}: {count} 个样本 ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def get_name(self) -> str:
        return "DBSCAN"


# ==================== 4. VQA聚类管道 ====================
class VQAClusteringPipeline:
    """VQA数据聚类管道"""
    
    def __init__(
        self,
        feature_extractor,  # VQAFeatureExtractor实例
        clustering_algorithm: ClusteringAlgorithm,
        cache_dir: str = "./vqa_cache",
        batch_size: int = 100,
        clustering_params: Dict = None
    ):
        self.feature_extractor = feature_extractor
        self.clustering_algorithm = clustering_algorithm
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.clustering_params = clustering_params or {}
        
        # 特征缓存
        self.feature_cache = FeatureCache(
            cache_dir, 
            feature_dim=self.feature_extractor.get_feature_names().__len__()
        )
    
    def run(self, input_jsonl: str, output_path: str):
        """执行完整聚类流程"""
        print("\n" + "=" * 70)
        print("🚀 VQA数据聚类流程")
        print("=" * 70)
        
        # 阶段1: 统计样本数
        print("\n[阶段 1/4] 统计数据...")
        n_samples = self._count_samples(input_jsonl)
        print(f"  📊 总样本数: {n_samples}")
        
        # 阶段2: 批量提取特征
        print("\n[阶段 2/4] 批量提取特征...")
        self._extract_features(input_jsonl, n_samples)
        
        # 阶段3: 执行聚类
        print("\n[阶段 3/4] 执行聚类...")
        labels = self._perform_clustering()
        
        # 阶段4: 保存结果
        print("\n[阶段 4/4] 保存结果...")
        self._save_results(input_jsonl, labels, output_path)
        
        # 清理缓存
        self.feature_cache.cleanup()
        
        print("\n" + "=" * 70)
        print("✅ 聚类完成!")
        print("=" * 70 + "\n")
    
    def _count_samples(self, data_path: str) -> int:
        """统计样本数"""
        loader = VQADataLoader(data_path, self.batch_size)
        return loader.count_samples()
    
    def _extract_features(self, data_path: str, n_samples: int):
        """批量提取特征并缓存"""
        # 创建缓存
        self.feature_cache.create_cache('vqa', n_samples)
        
        loader = VQADataLoader(data_path, self.batch_size)
        
        batch_idx = 0
        global_start_idx = 0
        
        for batch_samples in loader.stream_batches():
            batch_idx += 1
            batch_size = len(batch_samples)
            
            print(f"  批次 {batch_idx}: 处理 {batch_size} 个样本...")
            
            # 提取特征
            features = self.feature_extractor.extract_batch(batch_samples)
            
            # 写入缓存
            indices = np.arange(global_start_idx, global_start_idx + batch_size)
            self.feature_cache.write_batch('vqa', indices, features)
            
            global_start_idx += batch_size
            
            # 内存管理
            del features
            gc.collect()
        
        print(f"  ✅ 特征提取完成")
    
    def _perform_clustering(self) -> np.ndarray:
        """执行聚类"""
        # 从缓存读取所有特征
        print(f"  📥 从缓存加载特征...")
        features = self.feature_cache.read_all('vqa')
        print(f"  ✅ 加载完成: {features.shape}")
        
        # 执行聚类
        print(f"  🎯 使用算法: {self.clustering_algorithm.get_name()}")
        labels = self.clustering_algorithm.fit_predict(features, **self.clustering_params)
        
        # 评估聚类质量
        self._evaluate_clustering(features, labels)
        
        return labels
    
    def _evaluate_clustering(self, features: np.ndarray, labels: np.ndarray):
        """评估聚类质量"""
        print(f"\n  📊 聚类质量评估:")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # 排除噪声点
        
        if n_clusters <= 1:
            print(f"    ⚠️  只有1个簇,无法计算质量指标")
            return
        
        # 采样评估(大数据集)
        if len(features) > 10000:
            sample_size = 10000
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[sample_indices]
            labels_sample = labels[sample_indices]
        else:
            features_sample = features
            labels_sample = labels
        
        try:
            # 轮廓系数
            sil_score = silhouette_score(features_sample, labels_sample)
            print(f"    - Silhouette Score: {sil_score:.4f} (越接近1越好)")
            
            # Calinski-Harabasz指数
            ch_score = calinski_harabasz_score(features_sample, labels_sample)
            print(f"    - Calinski-Harabasz: {ch_score:.2f} (越大越好)")
            
            # Davies-Bouldin指数
            db_score = davies_bouldin_score(features_sample, labels_sample)
            print(f"    - Davies-Bouldin: {db_score:.4f} (越小越好)")
            
        except Exception as e:
            print(f"    ⚠️  评估失败: {e}")
    
    def _save_results(self, data_path: str, labels: np.ndarray, output_path: str):
        """保存聚类结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建聚类结果字典
        clusters = defaultdict(list)
        
        loader = VQADataLoader(data_path, self.batch_size)
        
        sample_idx = 0
        for batch_samples in loader.stream_batches():
            for sample in batch_samples:
                cluster_id = int(labels[sample_idx])
                
                clusters[cluster_id].append({
                    'sample_id': sample_idx,
                    'cluster_id': cluster_id,
                    'data': sample
                })
                
                sample_idx += 1
        
        # 格式化输出
        output_data = {
            'metadata': {
                'total_samples': len(labels),
                'n_clusters': len(clusters),
                'algorithm': self.clustering_algorithm.get_name()
            },
            'clusters': []
        }
        
        for cluster_id in sorted(clusters.keys()):
            samples = clusters[cluster_id]
            output_data['clusters'].append({
                'cluster_id': cluster_id,
                'size': len(samples),
                'percentage': len(samples) / len(labels) * 100,
                'samples': samples
            })
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 结果已保存: {output_path}")
        
        # 打印统计信息
        print(f"\n  📈 聚类统计:")
        for cluster_info in output_data['clusters']:
            print(f"    簇 {cluster_info['cluster_id']}: "
                  f"{cluster_info['size']} 样本 ({cluster_info['percentage']:.1f}%)")


# ==================== 5. 使用示例 ====================
if __name__ == "__main__":
    
    # 1. 初始化特征提取器
    print("🔧 初始化特征提取器...")
    
    # ========== 特征配置示例 ==========
    # 方案A: 使用所有特征（默认）
    # feature_config = None  # 或使用默认配置
    
    # 方案B: 只使用问题特征进行聚类
    feature_config = {
        'image': False,
        'question': False,    
        'answer': True,
        'statistical': False,
        'interaction': False
    }
    
    # 方案C: 使用问题和答案特征
    # feature_config = {
    #     'image': False,
    #     'question': True,
    #     'answer': True,
    #     'statistical': False,
    #     'interaction': False
    # }
    
    # 方案D: 使用图像和问题特征
    # feature_config = {
    #     'image': True,
    #     'question': True,
    #     'answer': False,
    #     'statistical': False,
    #     'interaction': True  # 交互特征需要图像和对话特征
    # }
    
    feature_extractor = VQAFeatureExtractor(
        model_name="openai/clip-vit-base-patch32",
        device="cuda",  # 或 "cpu"
        normalize=True,
        feature_config=feature_config  # 传入特征配置
    )
    
    # 2. 选择聚类算法
    # 方案A: 自动选择最优簇数的KMeans
    clustering_algo = AutoMiniBatchKMeans(random_state=42, batch_size=1000)
    clustering_params = {
        'n_clusters': None,  # None表示自动选择
        'min_clusters': 1,
        'max_clusters': 20,
        'auto_select': True
    }
    
    # 方案B: 指定簇数的KMeans
    # clustering_algo = AutoMiniBatchKMeans(random_state=42, batch_size=1000)
    # clustering_params = {'n_clusters': 10}
    
    # # 方案C: 层次聚类（自动选择最优簇数）
    # clustering_algo = HierarchicalClustering(linkage='ward')
    # clustering_params = {
    #     'n_clusters': None,  # None表示自动选择
    #     'min_clusters': 1,
    #     'max_clusters': 20,
    #     'auto_select': True
    # }
    
    # 方案C1: 层次聚类（指定簇数）
    # clustering_algo = HierarchicalClustering(linkage='ward')
    # clustering_params = {'n_clusters': 10}
    
    # 方案D: DBSCAN(基于密度，自动选择最优参数)
    # clustering_algo = DensityClustering(eps=0.5, min_samples=5)  # 初始值，会被自动优化
    # clustering_params = {
    #     'eps': None,  # None表示自动选择
    #     'min_samples': None,  # None表示自动选择
    #     'eps_range': (0.1, 2.0),  # eps搜索范围
    #     'eps_steps': 10,  # eps搜索步数
    #     'min_samples_range': (3, 10),  # min_samples搜索范围
    #     'auto_select': True
    # }
    
    # 方案D1: DBSCAN(指定参数)
    # clustering_algo = DensityClustering(eps=0.5, min_samples=5)
    # clustering_params = {'eps': 0.5, 'min_samples': 5}
    
    # 3. 初始化聚类管道
    pipeline = VQAClusteringPipeline(
        feature_extractor=feature_extractor,
        clustering_algorithm=clustering_algo,
        cache_dir="./vqa_cache",
        batch_size=100,
        clustering_params=clustering_params
    )
    
    # 4. 执行聚类
    pipeline.run(
        input_jsonl="/user/zhuxuzhou/a_cluster_test/converted_clean_content_standardized_final.jsonl",  #  "/user/zhuxuzhou/a_cluster_test/converted_clean_content.jsonl",
        output_path="/user/zhuxuzhou/a_whole_pipeline/cluster/vqa_clustered_results_final.json"
    )
    
    print("\n✨ 全部完成!")
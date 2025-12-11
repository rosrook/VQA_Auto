"""
三级模型数据质量评估器 (evaluator.py)

该模块实现了使用三个不同能力级别的模型(弱、正常、SOTA)对图片文字数据进行
质量评估和训练效果打分的功能。

理论依据:
1. 多模型集成评估 (Multi-Model Ensemble Evaluation):
   - 基于 Dietterich (2000) "Ensemble Methods in Machine Learning"
   - 通过不同能力级别的模型提供互补性评估

2. 数据质量评估框架:
   - 参考 Lin et al. (ECCV 2024) "VQAScore: Evaluating Text-to-Visual Models"
   - 参考 Dong et al. (2025) "SAIL-VL: Scalable Vision Language Model Training"
   
3. 弱学习器集成理论:
   - 基于 Freund & Schapire (1997) "A Decision-Theoretic Generalization of Boosting"
   - 多个弱模型的集成可以达到强模型的效果

4. Vision-Language模型评估:
   - 参考 NaturalBench (NeurIPS 2024) 和 VQAScore 的评估方法
"""

import os
import json
import logging
import asyncio
import multiprocessing
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
# import numpy as np  # 暂时注释，未使用
from pathlib import Path
import base64
from PIL import Image
import io
from typing import Literal

os.environ['MULTIPROCESSING_METHOD'] = 'spawn'
os.environ['PYTHONMULTIPROCESSINGMETHOD'] = 'spawn'

# 设置 PyTorch 相关环境变量以节省内存（可通过命令行覆盖）
# 关闭 CUDA 图优化（如果环境变量未设置，则默认关闭）
if 'PYTORCH_DISABLE_CUDA_GRAPH' not in os.environ:
    os.environ.setdefault('PYTORCH_DISABLE_CUDA_GRAPH', '1')
# 关闭 cuDNN 基准测试（节省内存）
os.environ.setdefault('TORCH_CUDNN_BENCHMARK', '0')
# 使用更保守的 CUDA 内存分配策略
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:False')

from .mllm_client import QwenVLClientVLLM

# 设置 multiprocessing start method 为 'spawn'，解决 vllm CUDA 初始化问题
# 必须在导入 vllm 之前设置
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass

# Pydantic imports (用于数据验证)
try:
    from pydantic import BaseModel, Field
except ImportError:
    # 如果pydantic不可用，尝试使用langchain的pydantic
    try:
        from langchain_core.pydantic_v1 import BaseModel, Field
    except ImportError:
        raise ImportError(
            "需要安装 pydantic 或 langchain。请运行: pip install pydantic 或 pip install langchain"
        )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """模型能力级别"""
    WEAK = "weak"           # 较弱模型 (如 SmolVLM-500M, Gemma-2B)
    NORMAL = "normal"       # 正常模型 (如 LLaVA-7B, InternVL-8B)
    SOTA = "sota"          # SOTA模型 (如 GPT-4V, Claude-3.5-Sonnet, Gemini-Pro)


@dataclass
class QualityMetric:
    """数据质量评估指标（已废弃，保留用于兼容性）"""
    name: str
    description: str
    score: float  # 0-10分
    confidence: float  # 置信度 0-1
    rationale: str  # 评分理由


@dataclass
class DataSample:
    """数据样本"""
    sample_id: str
    # 支持多张图片的base64编码
    image_base64: Optional[List[str]] = None
    dialogue: Optional[List[Dict[str, str]]] = None  # List of {question, answer} dicts
    metadata: Optional[Dict] = None


@dataclass
class EvaluationResult:
    """单个模型的评估结果"""
    model_name: str
    model_capability: ModelCapability
    sample_id: str
    
    # 正确性判断
    correctness_score: float  # 0-10分
    correctness_confidence: float
    correctness_rationale: str
    
    # 训练效果评分
    training_quality_score: float  # 0-10分
    training_quality_confidence: float
    training_quality_rationale: str
    
    # 推荐的筛选指标
    # 每个字典包含两个字段：
    # - "general_metric_description": 抽象的概括性标准（如"视觉识别中空间识别正确性"）
    # - "specific_metric_description": 具体贴近数据例子的具体标准（如"待检测数据中answer是否正确判断了物体所在位置"）
    recommended_metrics: List[Dict[str, str]]
    
    # 综合评估
    overall_assessment: str
    timestamp: str
    
    # 测试模式：信息复述
    information_recap: Optional[str] = None  # 模型对接收到的信息的复述


class ModelConfig(BaseModel):
    """模型配置的Pydantic模型"""
    name: str = Field(description="模型名称")
    capability: str = Field(description="模型能力级别")
    model_type: Literal["api", "local", "huggingface"] = Field(default="api", description="模型加载类型: api/local/huggingface")
    
    # API相关配置
    api_endpoint: Optional[str] = Field(default=None, description="API端点")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    
    # 本地模型相关配置
    local_model_path: Optional[str] = Field(default=None, description="本地模型路径")
    local_device: Optional[str] = Field(default="cuda", description="本地模型设备: cuda/cpu")
    
    # HuggingFace相关配置
    hf_model_id: Optional[str] = Field(default=None, description="HuggingFace模型ID")
    hf_device_map: Optional[str] = Field(default="auto", description="HuggingFace设备映射")
    hf_trust_remote_code: bool = Field(default=False, description="是否信任远程代码")
    
    # 通用配置
    max_tokens: int = Field(default=2000, description="最大token数")
    temperature: float = Field(default=0.1, description="温度参数")


class DataQualityEvaluator:
    """
    数据质量评估器
    
    使用三级模型(弱、正常、SOTA)对图片文字数据进行评估
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化评估器
        
        参数:
            config_path: 配置文件路径
        """
        self.models = self._load_model_configs(config_path)
        self.evaluation_history = []
        # 存储已加载的模型实例（用于本地和HuggingFace模型）
        self.loaded_models: Dict[str, Any] = {}
        
    def _load_model_configs(self, config_path: Optional[str]) -> Dict[str, ModelConfig]:
        """加载模型配置"""
        logger.info(f"Loading model configs, config_path: {config_path}")
        
        # 默认配置（使用API方式）
        default_configs = {
            ModelCapability.WEAK: ModelConfig(
                name="smolvlm-500m",
                capability="weak",
                model_type="api",
                api_endpoint="https://api.anthropic.com/v1/messages",
                max_tokens=1500
            ),
            ModelCapability.NORMAL: ModelConfig(
                name="internvl-8b",
                capability="normal",
                model_type="api",
                api_endpoint="https://api.anthropic.com/v1/messages",
                max_tokens=2000
            ),
            ModelCapability.SOTA: ModelConfig(
                name="claude-sonnet-4-20250514",
                capability="sota",
                model_type="api",
                api_endpoint="https://api.anthropic.com/v1/messages",
                max_tokens=3000
            )
        }
        
        logger.info("Default configs initialized:")
        for cap, cfg in default_configs.items():
            logger.info(f"  {cap.value}: {cfg.name} (type={cfg.model_type}, endpoint={cfg.api_endpoint}, api_key={'set' if cfg.api_key else 'NOT SET'})")
        
        if config_path:
            # 尝试解析路径（相对路径或绝对路径）
            if not os.path.isabs(config_path):
                # 相对路径：尝试相对于当前工作目录和脚本目录
                current_dir = os.getcwd()
                script_dir = os.path.dirname(os.path.abspath(__file__))
                possible_paths = [
                    os.path.join(current_dir, config_path),
                    os.path.join(script_dir, config_path),
                    os.path.join(script_dir, "..", config_path),
                ]
                logger.info(f"Config path is relative, trying: {possible_paths}")
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        config_path = abs_path
                        logger.info(f"Found config file at: {config_path}")
                        break
                else:
                    logger.warning(f"Config file not found in any of: {possible_paths}")
            else:
                logger.info(f"Config path is absolute: {config_path}")
            
            if os.path.exists(config_path):
                logger.info(f"Loading config from: {config_path}")
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    logger.info(f"Config file loaded successfully, keys: {list(config_data.keys())}")
                    
                    # 从model_configs中读取配置
                    if "model_configs" in config_data:
                        custom_configs = config_data["model_configs"]
                        logger.info(f"Found model_configs with keys: {list(custom_configs.keys())}")
                        
                        for cap, config in custom_configs.items():
                            if cap in ["weak", "normal", "sota"]:
                                logger.info(f"Processing {cap} model config: {config}")
                                try:
                                    capability = ModelCapability(cap)
                                    model_config = ModelConfig(**config)
                                    default_configs[capability] = model_config
                                    logger.info(f"Successfully loaded {cap} config: name={model_config.name}, type={model_config.model_type}, api_key={'set' if model_config.api_key else 'NOT SET'}")
                                except Exception as e:
                                    logger.error(f"Failed to load {cap} config: {e}")
                                    import traceback
                                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"No 'model_configs' key found in config file")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse config JSON: {e}")
                except Exception as e:
                    logger.error(f"Failed to load config file: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"Config file does not exist: {config_path}")
        else:
            logger.warning("No config_path provided, using default configs")
        
        # 输出最终配置
        logger.info("Final loaded configs:")
        for cap, cfg in default_configs.items():
            logger.info(f"  {cap.value}: {cfg.name} (type={cfg.model_type}, endpoint={cfg.api_endpoint}, api_key={'set' if cfg.api_key else 'NOT SET'})")
        
        return default_configs
    
    def _create_evaluation_prompt(
        self,
        sample: DataSample,
        model_capability: ModelCapability,
        use_simple_evaluation: bool = False,
        test_mode: bool = False
    ) -> str:
        """
        创建评估提示词
        
        根据模型能力级别调整提示词复杂度
        
        参数:
            sample: 数据样本
            model_capability: 模型能力级别
            use_simple_evaluation: 是否使用简单版评估（True=简单版，False=复杂版）
        """
        base_prompt = f"""You are a professional data quality evaluation expert. Please conduct a comprehensive evaluation of the following data sample.

Sample ID: {sample.sample_id}
"""
        
        if sample.dialogue:
            base_prompt += "\nDialogue Content:\n"
            for i, turn in enumerate(sample.dialogue, 1):
                question = turn.get('question', '')
                answer = turn.get('answer', '')
                base_prompt += f"Turn {i}:\n"
                base_prompt += f"  Question: {question}\n"
                base_prompt += f"  Answer: {answer}\n"
        
        if sample.image_base64:
            # 支持多张图片
            img_count = len(sample.image_base64) if isinstance(sample.image_base64, list) else 1
            base_prompt += f"\n[Image(s) provided: {img_count}]\n"
        
        # 根据模型能力调整任务复杂度
        if model_capability == ModelCapability.WEAK:
            if use_simple_evaluation:
                # Simple version: only one sentence asking for scores
                task_prompt = """
Please complete the following evaluation tasks:

1. **Correctness Assessment** (0-10 points): Please score the correctness of this data sample.

2. **Training Quality Assessment** (0-10 points): Please score the quality of this data sample as a training example.

3. **Key Metric Identification**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "spatial recognition correctness", "semantic understanding accuracy")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer correctly identifies the object's location")
   
   **Example**:
   - General metric description: "spatial recognition correctness"
   - Specific metric description: "whether the answer correctly identifies the object's location"
"""
            else:
                # Complex version: original detailed prompt
                task_prompt = """
Please complete the following basic evaluation tasks:

1. **Correctness Assessment** (0-10 points):
   - Evaluate the accuracy and reasonableness of the data
   - Explain your scoring rationale

2. **Training Quality Assessment** (0-10 points):
   - Evaluate the quality of this data as a training sample
   - Consider the data's representativeness and information content

3. **Key Metric Identification**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "spatial recognition correctness", "semantic understanding accuracy")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer correctly identifies the object's location")
   
   **Example**:
   - General metric description: "spatial recognition correctness"
   - Specific metric description: "whether the answer correctly identifies the object's location"
"""
        
        elif model_capability == ModelCapability.NORMAL:
            if use_simple_evaluation:
                # Simple version: only one sentence asking for scores
                task_prompt = """
Please complete the following evaluation tasks:

1. **Correctness Assessment** (0-10 points): Please score the correctness of this data sample.

2. **Training Quality Assessment** (0-10 points): Please score the quality of this data sample as a training example.

3. **Key Metric Identification**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "semantic understanding accuracy", "image-text alignment")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer accurately understands the question's inquiry about image content")
   
   **Example**:
   - General metric description: "semantic understanding accuracy"
   - Specific metric description: "whether the answer accurately understands the question's inquiry about image content"
"""
            else:
                # Complex version: original detailed prompt
                task_prompt = """
Please complete the following detailed evaluation tasks:

1. **Correctness Assessment** (0-10 points):
   - Data accuracy
   - Image-text consistency (if applicable)
   - Logical reasonableness
   Provide detailed scoring rationale and confidence level

2. **Training Quality Assessment** (0-10 points):
   - Data representativeness
   - Information richness
   - Sample diversity
   - Potential training value
   Provide detailed scoring rationale and confidence level

3. **Key Metric Identification**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "semantic understanding accuracy", "image-text alignment")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer accurately understands the question's inquiry about image content")
   
   **Example**:
   - General metric description: "semantic understanding accuracy"
   - Specific metric description: "whether the answer accurately understands the question's inquiry about image content"
"""
        
        else:  # SOTA
            if use_simple_evaluation:
                # Simple version: only one sentence asking for scores
                task_prompt = """
Please complete the following evaluation tasks as a domain expert:

1. **Correctness Assessment** (0-10 points): Please score the correctness of this data sample.

2. **Training Quality Assessment** (0-10 points): Please score the quality of this data sample as a training example.

3. **Systematic Metric System Construction**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "spatial relationship understanding accuracy", "cross-modal alignment")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer correctly describes the spatial position relationship of objects inquired in the question")
   
   **Example**:
   - General metric description: "spatial relationship understanding accuracy"
   - Specific metric description: "whether the answer correctly describes the spatial position relationship of objects inquired in the question"

4. **Overall Assessment**:
   Provide a comprehensive evaluation of this sample, including:
   - Strengths and weaknesses
   - Applicable scenarios
   - Improvement suggestions
"""
            else:
                # Complex version: original detailed prompt
                task_prompt = """
Please complete the following comprehensive evaluation tasks as a domain expert:

1. **In-Depth Correctness Assessment** (0-10 points):
   - Data accuracy (semantic, factual, logical)
   - Multimodal consistency (image-text alignment, cross-modal semantics)
   - Potential biases or errors
   - Edge case analysis
   Provide detailed scoring rationale, confidence level, and uncertainty analysis

2. **Comprehensive Training Quality Assessment** (0-10 points):
   - Data representativeness and coverage
   - Information density and complexity
   - Sample difficulty and challenge level
   - Expected contribution to model training
   - Potential negative impacts
   Provide detailed scoring rationale, confidence level, and risk assessment

3. **Systematic Metric System Construction**:
   Please carefully observe the current data sample and extract at most 3 key data filtering metrics from it. These metrics must be observed and extracted from the current data.
   
   **Important Understanding**:
   - If the current data quality is good (high score), the metrics you propose should be: **the key features observed from the current data that explain why it is "good"**, which can be used to filter similar high-quality data.
   - If the current data quality is poor (low score), the metrics you propose should be: **the key issues observed from the current data that explain why it is "poor"**, which can be used to filter out similar poor-quality data.
   
   **Format Requirements** (each metric contains two fields):
   - **General Metric Description** (general_metric_description): Summarize with a few phrases, keep it concise (e.g., "spatial relationship understanding accuracy", "cross-modal alignment")
   - **Specific Metric Description** (specific_metric_description): No more than one short sentence, specifically describe how this metric is reflected in the current data (e.g., "whether the answer correctly describes the spatial position relationship of objects inquired in the question")
   
   **Example**:
   - General metric description: "spatial relationship understanding accuracy"
   - Specific metric description: "whether the answer correctly describes the spatial position relationship of objects inquired in the question"

4. **Overall Assessment**:
   Provide a comprehensive evaluation of this sample, including:
   - Strengths and weaknesses
   - Applicable scenarios
   - Improvement suggestions
"""
        
        # 测试模式：添加信息复述要求
        test_mode_section = ""
        if test_mode:
            test_mode_section = """
    
**TEST MODE - Information Recap**:
Before providing your evaluation, please first recap the information you received:
- Briefly summarize the dialogue content (if any)
- Describe what you see in the image(s) (if any)
- Confirm the sample ID
This helps verify that you correctly received all the input information.
"""
        
        # 根据测试模式调整输出格式
        if test_mode:
            output_format = f"""
Please return the evaluation results in JSON format:
{{
    "information_recap": "<brief recap of the information you received - dialogue, images, sample ID>",
    "correctness_score": <float between 0-10>,
    "correctness_confidence": <float between 0-1>,
    "correctness_rationale": "<detailed rationale>",
    "training_quality_score": <float between 0-10>,
    "training_quality_confidence": <float between 0-1>,
    "training_quality_rationale": "<detailed rationale>",
    "recommended_metrics": [
        {{
            "general_metric_description": "<abstract general standard applicable to similar data>",
            "specific_metric_description": "<specific standard closely tied to the current data example, directly usable for judging sample quality>"
        }}
    ],
    "overall_assessment": "<comprehensive assessment>"
}}
"""
        else:
            output_format = """
Please return the evaluation results in JSON format:
{
    "correctness_score": <float between 0-10>,
    "correctness_confidence": <float between 0-1>,
    "correctness_rationale": "<detailed rationale>",
    "training_quality_score": <float between 0-10>,
    "training_quality_confidence": <float between 0-1>,
    "training_quality_rationale": "<detailed rationale>",
    "recommended_metrics": [
        {
            "general_metric_description": "<abstract general standard applicable to similar data>",
            "specific_metric_description": "<specific standard closely tied to the current data example, directly usable for judging sample quality>"
        }
    ],
    "overall_assessment": "<comprehensive assessment>"
}
"""
        
        return base_prompt + task_prompt + test_mode_section + output_format
    
    def _load_local_model(self, model_config: ModelConfig):
        """加载本地模型（使用 QwenVLClientVLLM 或 transformers）"""
        model_key = f"{model_config.capability}_{model_config.name}"
        
        # 检查是否已经加载
        if model_key in self.loaded_models:
            logger.info(f"Model {model_key} already loaded, reusing")
            return self.loaded_models[model_key]
        
        try:
            logger.info(f"Loading local model from {model_config.local_model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(model_config.local_model_path):
                logger.error(f"Model path does not exist: {model_config.local_model_path}")
                return None
            
            # 确定设备
            import torch
            device = model_config.local_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            
            # 检查是否是 MoE 模型（MoE 模型 vllm 支持不好，容易出现 CUDA fork 错误）
            is_moe = "moe" in model_config.name.lower() or "moe" in model_config.local_model_path.lower() or "A3B" in model_config.name
            
            if is_moe:
                logger.warning(
                    f"MoE model detected ({model_config.name}). "
                    "vLLM has known issues with MoE models (CUDA fork errors). "
                    "Skipping vLLM and using transformers instead."
                )
                # 对于 MoE 模型，直接使用 transformers，跳过 vllm
                return self._load_local_model_with_transformers(model_config, device)
            
            # 非 MoE 模型，尝试使用 vllm
            try:
                # 确定 tensor_parallel_size（根据 GPU 数量，但限制最大值避免资源问题）
                tensor_parallel_size = 1
                if device == "cuda" and torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                        # 充分利用所有可用的 GPU（对于 8 个 GPU，使用 tensor_parallel_size=8）
                    tensor_parallel_size = available_gpus
                    logger.info(f"Detected {available_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size} for vLLM")
                    logger.info(f"This will distribute the model across all {available_gpus} GPU(s) using tensor parallelism")
                
                # 设置 max_model_len（上下文窗口大小，应该足够大以容纳输入和输出）
                max_model_len = max(model_config.max_tokens * 4, 8192) if model_config.max_tokens else 8192
                
                # 创建 QwenVLClientVLLM 客户端
                logger.info(f"Initializing QwenVLClientVLLM with tensor_parallel_size={tensor_parallel_size}, max_model_len={max_model_len}")
                client = QwenVLClientVLLM(
                    model_path=model_config.local_model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                    max_model_len=max_model_len,
                    max_num_seqs=256,  # 最大并发序列数
                    temperature=model_config.temperature,
                    top_p=0.95,  # 默认 top_p
                    max_tokens=model_config.max_tokens,
                    dtype="bfloat16" if device == "cuda" else "float32",
                    trust_remote_code=True,
                )
                
                logger.info(f"Successfully loaded local model with vLLM: {model_config.name}")
                
                # 保存到 loaded_models
                self.loaded_models[model_key] = {
                    "client": client,
                    "device": device,
                    "model_type": "vl",
                    "use_mllm_client": True
                }
                
                return self.loaded_models[model_key]
                
            except (ImportError, RuntimeError, Exception) as vllm_error:
                error_msg = str(vllm_error)
                if "Cannot re-initialize CUDA in forked subprocess" in error_msg or "spawn" in error_msg.lower():
                    logger.warning(
                        f"vLLM failed with CUDA multiprocessing error: {vllm_error}. "
                        "Falling back to transformers."
                    )
                else:
                    logger.warning(f"vLLM loading failed: {vllm_error}. Falling back to transformers.")
                
                # 清理 GPU 缓存（vllm 失败后可能占用了内存）
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU cache after vLLM failure")
                
                # 回退到 transformers
                return self._load_local_model_with_transformers(model_config, device)
            
        except Exception as e:
            logger.error(f"Failed to load local model: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _load_local_model_with_transformers(self, model_config: ModelConfig, device: str):
        """使用 transformers 加载本地模型（用于 MoE 模型或 vllm 失败时的回退）"""
        model_key = f"{model_config.capability}_{model_config.name}"
        
        try:
            logger.info(f"Loading model with transformers: {model_config.name}")
            
            from transformers import AutoProcessor
            import torch
            
            # 清理 GPU 缓存，确保有足够内存
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 获取 GPU 数量和内存信息
                available_gpus = torch.cuda.device_count()
                logger.info(f"Detected {available_gpus} GPU(s) for transformers loading")
                
                # 获取每个 GPU 的内存信息
                for gpu_id in range(available_gpus):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                    logger.info(f"GPU {gpu_id} memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
            
            # 关闭梯度计算以节省内存（推理模式）
            torch.set_grad_enabled(False)
            
            # 为所有 GPU 设置 max_memory（如果使用 CUDA）
            max_memory_dict = None
            if device == "cuda" and torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                # 为每个 GPU 设置内存限制
                # 根据实际 GPU 内存大小调整：44.53 GB total，设置 40GB 作为保守估计
                # 如果 GPU 内存更大，可以相应调整
                gpu_memory_per_device = "40GiB"  # 可以根据实际情况调整
                max_memory_dict = {i: gpu_memory_per_device for i in range(available_gpus)}
                logger.info(f"Setting max_memory for {available_gpus} GPU(s): {max_memory_dict}")
                
                # 检查是否安装了 accelerate
                try:
                    import accelerate
                    logger.info("accelerate library is available, will use device_map='auto' for multi-GPU distribution")
                except ImportError:
                    logger.warning(
                        "accelerate library not found. device_map='auto' will fail.\n"
                        "For proper multi-GPU support, please install: pip install accelerate\n"
                        "Without accelerate, model may only use a single GPU (may cause OOM for large models)"
                    )
            
            processor = AutoProcessor.from_pretrained(
                model_config.local_model_path,
                trust_remote_code=True
            )
            
            # 尝试使用正确的模型类（Qwen3VLMoeForConditionalGeneration）
            try:
                from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
                logger.info("Using Qwen3VLMoeForConditionalGeneration for MoE model (from specific module)")
                
                # 对于大模型，使用 device_map="auto" 让 transformers 自动分配（需要 accelerate）
                # 如果没有 accelerate，则手动加载到设备
                try:
                    # 使用 torch.inference_mode() 上下文管理器（比 no_grad 更节省内存）
                    with torch.inference_mode():
                        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                            model_config.local_model_path,
                            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                            device_map="auto" if device == "cuda" else None,  # 自动分配 GPU 内存到所有可用 GPU
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,  # 减少 CPU 内存使用
                            max_memory=max_memory_dict,  # 为所有 GPU 设置内存限制
                        )
                    # 如果使用 device_map="auto"，不需要手动 .to(device)
                    if device == "cpu" or (device == "cuda" and hasattr(model, 'device') and model.device.type != "cuda"):
                        model = model.to(device)
                    # 设置为评估模式（关闭 dropout 等，节省内存）
                    model.eval()
                except Exception as device_map_error:
                    # 如果 device_map 失败（可能没有 accelerate），尝试手动多 GPU 分布
                    error_msg = str(device_map_error)
                    if "accelerate" in error_msg.lower():
                        logger.error(
                            f"device_map='auto' requires 'accelerate' library.\n"
                            f"Please install it: pip install accelerate\n"
                            f"Attempting to load model without device_map (may cause OOM on single GPU)..."
                        )
                    
                    # 尝试手动创建 device_map 字典来分布模型
                    if device == "cuda" and torch.cuda.is_available():
                        available_gpus = torch.cuda.device_count()
                        logger.info(f"Attempting manual multi-GPU distribution across {available_gpus} GPU(s)")
                        
                        # 对于 MoE 模型，尝试手动分布
                        # 注意：这是一个简化的方案，可能不如 accelerate 的自动分布效果好
                        try:
                            with torch.inference_mode():
                                # 先尝试不使用 device_map，但设置 max_memory
                                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                                    model_config.local_model_path,
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True,
                                    max_memory=max_memory_dict,
                                    # 不设置 device_map，让模型先加载到 CPU
                                )
                            
                            # 手动将模型分布到多个 GPU（简化版：只移动主要部分）
                            # 注意：完整的模型并行需要更复杂的逻辑
                            logger.warning(
                                "Manual multi-GPU distribution is limited. "
                                "For better performance, please install accelerate: pip install accelerate"
                            )
                            # 暂时只移动到 GPU 0，但至少不会因为 device_map 失败而崩溃
                            model = model.to(f"cuda:0")
                            model.eval()
                        except Exception as manual_error:
                            logger.error(f"Manual device placement also failed: {manual_error}")
                            logger.error("Please install accelerate for proper multi-GPU support: pip install accelerate")
                            raise
                    else:
                        # CPU 模式
                        with torch.inference_mode():
                            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                                model_config.local_model_path,
                                torch_dtype=torch.float32,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                            )
                        model.eval()
            except (ImportError, AttributeError):
                # 尝试从顶层导入
                try:
                    from transformers import Qwen3VLMoeForConditionalGeneration
                    logger.info("Using Qwen3VLMoeForConditionalGeneration for MoE model (from top-level)")
                    try:
                        with torch.inference_mode():
                            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                                model_config.local_model_path,
                                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                device_map="auto" if device == "cuda" else None,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                max_memory=max_memory_dict,  # 为所有 GPU 设置内存限制
                            )
                        if device == "cpu" or (device == "cuda" and hasattr(model, 'device') and model.device.type != "cuda"):
                            model = model.to(device)
                        model.eval()
                    except Exception as device_map_error:
                        error_msg = str(device_map_error)
                        if "accelerate" in error_msg.lower():
                            logger.error(
                                f"device_map='auto' requires 'accelerate' library.\n"
                                f"Please install it: pip install accelerate"
                            )
                        logger.warning(f"device_map='auto' failed: {device_map_error}, falling back to manual device placement")
                        # 注意：没有 accelerate 时，模型只能加载到单个 GPU，可能导致 OOM
                        logger.warning("Without accelerate, model will be loaded to a single GPU (GPU 0), which may cause OOM for large models")
                        with torch.inference_mode():
                            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                                model_config.local_model_path,
                                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                # 不设置 max_memory，因为 device_map 不可用
                            )
                        # 只能移动到单个 GPU
                        model = model.to(f"cuda:0" if device == "cuda" else device)
                        model.eval()
                except (ImportError, AttributeError):
                    # 如果无法导入，尝试使用 AutoModel
                    logger.warning("Qwen3VLMoeForConditionalGeneration not available, trying AutoModel")
                    from transformers import AutoModel
                    try:
                        with torch.inference_mode():
                            model = AutoModel.from_pretrained(
                                model_config.local_model_path,
                                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                device_map="auto" if device == "cuda" else None,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                max_memory=max_memory_dict,  # 为所有 GPU 设置内存限制
                            )
                        if device == "cpu" or (device == "cuda" and hasattr(model, 'device') and model.device.type != "cuda"):
                            model = model.to(device)
                        model.eval()
                    except Exception as device_map_error:
                        error_msg = str(device_map_error)
                        if "accelerate" in error_msg.lower():
                            logger.error(
                                f"device_map='auto' requires 'accelerate' library.\n"
                                f"Please install it: pip install accelerate"
                            )
                        logger.warning(f"device_map='auto' failed: {device_map_error}, falling back to manual device placement")
                        logger.warning("Without accelerate, model will be loaded to a single GPU (GPU 0), which may cause OOM for large models")
                        with torch.inference_mode():
                            model = AutoModel.from_pretrained(
                                model_config.local_model_path,
                                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                            )
                        model = model.to(f"cuda:0" if device == "cuda" else device)
                        model.eval()
            
            # 再次清理 GPU 缓存（加载后）
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_gpus = torch.cuda.device_count()
                
                # 检查模型实际分布在哪些 GPU 上
                if hasattr(model, 'hf_device_map'):
                    device_map = model.hf_device_map
                    logger.info(f"Model device map: {device_map}")
                    # 统计使用的 GPU
                    used_gpus = set()
                    for module_name, device_info in device_map.items():
                        if isinstance(device_info, (int, str)):
                            if isinstance(device_info, int):
                                used_gpus.add(device_info)
                            elif 'cuda:' in str(device_info):
                                gpu_id = int(str(device_info).split(':')[1])
                                used_gpus.add(gpu_id)
                    logger.info(f"Model is distributed across {len(used_gpus)} GPU(s): {sorted(used_gpus)}")
                elif hasattr(model, 'device'):
                    logger.info(f"Model is on device: {model.device}")
                
                # 显示每个 GPU 的内存使用情况
                for gpu_id in range(available_gpus):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                    logger.info(f"GPU {gpu_id} memory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
            logger.info(f"Successfully loaded model with transformers: {model_config.name}")
            
            # 保存到 loaded_models
            self.loaded_models[model_key] = {
                "model": model,
                "processor": processor,
                "device": device,
                "model_type": "vl",
                "use_mllm_client": False
            }
            
            return self.loaded_models[model_key]
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory when loading model: {e}")
            if device == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.error(f"GPU memory status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
                logger.error("Try: 1) Clear GPU cache, 2) Use smaller model, 3) Use CPU, 4) Reduce max_memory limit")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"Failed to load model with transformers: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _load_huggingface_model(self, model_config: ModelConfig):
        """从HuggingFace加载模型"""
        model_key = f"{model_config.capability}_{model_config.name}"
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        try:
            logger.info(f"Loading HuggingFace model: {model_config.hf_model_id}")
            # 这里需要根据实际模型类型加载
            # 示例：使用transformers加载
            """
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            import torch
            
            device_map = model_config.hf_device_map
            
            if "vision" in model_config.hf_model_id.lower() or "vl" in model_config.hf_model_id.lower():
                # Vision-Language模型
                processor = AutoProcessor.from_pretrained(
                    model_config.hf_model_id,
                    trust_remote_code=model_config.hf_trust_remote_code
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.hf_model_id,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=model_config.hf_trust_remote_code
                )
                self.loaded_models[model_key] = {"model": model, "processor": processor}
            else:
                # 纯语言模型
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_model_id,
                    trust_remote_code=model_config.hf_trust_remote_code
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.hf_model_id,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=model_config.hf_trust_remote_code
                )
                self.loaded_models[model_key] = {"model": model, "tokenizer": tokenizer}
            """
            # 暂时返回None，实际使用时需要实现
            logger.warning("HuggingFace model loading not fully implemented, using mock response")
            self.loaded_models[model_key] = None
            return None
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            return None
    
    async def _call_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        image_data: Optional[List[str]] = None
    ) -> Dict:
        """
        调用模型（支持API、本地模型、HuggingFace三种方式）
        """
        try:
            # 根据模型类型选择调用方式
            if model_config.model_type == "api":
                return await self._call_api_model(model_config, prompt, image_data)
            elif model_config.model_type == "local":
                return await self._call_local_model(model_config, prompt, image_data)
            elif model_config.model_type == "huggingface":
                return await self._call_huggingface_model(model_config, prompt, image_data)
            else:
                logger.error(f"Unknown model type: {model_config.model_type}")
                return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return self._generate_fallback_response()
    
    # async def _call_api_model(
    #     self,
    #     model_config: ModelConfig,
    #     prompt: str,
    #     image_data: Optional[List[str]] = None
    # ) -> Dict:
    #     """调用API模型（参考 api_client.py 的实现方式）"""
    #     # 记录请求信息（用于错误诊断）
    #     request_info = {
    #         "model": model_config.name,
    #         "endpoint": model_config.api_endpoint,
    #         "prompt_length": len(prompt),
    #         "image_count": len(image_data) if image_data else 0,
    #         "max_tokens": model_config.max_tokens,
    #         "temperature": model_config.temperature
    #     }
        
    #     try:
    #         from openai import AsyncOpenAI
    #         from PIL import Image
    #         from io import BytesIO
    #         import base64
            
    #         logger.info(f"Calling API model {model_config.name} for evaluation...")
    #         logger.debug(f"API endpoint: {model_config.api_endpoint}")
    #         logger.debug(f"Prompt length: {len(prompt)} chars, Images: {len(image_data) if image_data else 0}")
    #         logger.debug(f"Request parameters: max_tokens={model_config.max_tokens}, temperature={model_config.temperature}")
            
    #         # 构建消息内容（参考 api_client.py 的 _build_messages 方法）
    #         content = []
            
    #         if image_data:
    #             # 处理图片：将 base64 字符串转换为 Image 对象，然后重新编码（与 api_client.py 保持一致）
    #             for img_b64 in image_data:
    #                 try:
    #                     # 解码 base64 字符串
    #                     img_bytes = base64.b64decode(img_b64)
    #                     # 转换为 PIL Image
    #                     img = Image.open(BytesIO(img_bytes))
    #                     # 确保是 RGB 模式
    #                     if img.mode != 'RGB':
    #                         img = img.convert('RGB')
                        
    #                     # 重新编码为 base64（与 api_client.py 保持一致）
    #                     buffered = BytesIO()
    #                     img.save(buffered, format="JPEG", quality=85)
    #                     img_b64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
    #                     content.append({
    #                         "type": "image_url",
    #                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64_encoded}"}
    #                     })
    #                 except Exception as e:
    #                     logger.warning(f"Failed to process image: {e}")
    #                     # 如果处理失败，直接使用原始 base64
    #                     content.append({
    #                         "type": "image_url",
    #                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
    #                     })
            
    #         # 添加文本提示
    #         content.append({"type": "text", "text": prompt})
            
    #         # 构建消息（与 api_client.py 保持一致）
    #         messages = [{"role": "user", "content": content}]
            
    #         # 验证 API key 是否存在
    #         if not model_config.api_key or model_config.api_key == "NOT SET" or len(model_config.api_key.strip()) == 0:
    #             error_msg = (
    #                 f"API key is not set for model {model_config.name}\n"
    #                 f"Request info: {request_info}\n"
    #                 f"Please check config.json and ensure api_key is set"
    #             )
    #             logger.error(error_msg)
    #             return self._generate_fallback_response()
            
    #         # 按照正确示例创建客户端
    #         # config中的api_endpoint就是base_url，直接使用
    #         base_url = model_config.api_endpoint.rstrip('/') if model_config.api_endpoint else None
    #         if not base_url:
    #             logger.error(f"API endpoint is not set for model {model_config.name}")
    #             return self._generate_fallback_response()
            
    #         # 如果base_url不以/v1结尾，自动添加（兼容旧配置）
    #         if not base_url.endswith('/v1'):
    #             base_url = base_url + '/v1'
    #             logger.debug(f"自动添加 /v1 路径，使用 base_url: {base_url}")
            
    #         logger.debug(f"Using base_url: {base_url}")
            
    #         # 创建OpenAI异步客户端（按照用户提供的正确示例）
    #         client = AsyncOpenAI(
    #             api_key=model_config.api_key,
    #             base_url=base_url
    #         )
            
    #         logger.info(f"Sending request to {model_config.name}...")
            
    #         # 调用API（按照用户提供的正确示例格式）
    #         response = await client.chat.completions.create(
    #             model=model_config.name,
    #             messages=messages,
    #             max_tokens=model_config.max_tokens,
    #             temperature=model_config.temperature
    #         )
            
    #         result_text = response.choices[0].message.content
    #         logger.info(f"API call successful, response length: {len(result_text)} chars")
            
    #         # 记录token使用情况（参考 api_client.py）
    #         if hasattr(response, 'usage') and response.usage:
    #             logger.info(f"Token usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
            
    #         # 解析JSON响应
    #         import re
    #         json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    #         if json_match:
    #             try:
    #                 result_dict = json.loads(json_match.group())
    #                 logger.debug("Successfully parsed JSON response")
    #             except json.JSONDecodeError as e:
    #                 logger.warning(f"Failed to parse JSON from response: {e}")
    #                 logger.warning(f"Response text (first 500 chars): {result_text[:500]}")
    #                 # 尝试解析整个响应
    #                 try:
    #                     result_dict = json.loads(result_text)
    #                 except:
    #                     logger.error("Failed to parse response as JSON, using fallback")
    #                     return self._generate_fallback_response()
    #         else:
    #             # 尝试直接解析整个响应
    #             try:
    #                 result_dict = json.loads(result_text)
    #             except json.JSONDecodeError as e:
    #                 logger.error(f"Response is not valid JSON: {e}")
    #                 logger.error(f"Response text (first 500 chars): {result_text[:500]}")
    #                 return self._generate_fallback_response()
            
    #         return result_dict
            
    #     except ImportError as e:
    #         error_msg = (
    #             f"Missing required library: {e}\n"
    #             f"Please install: pip install openai\n"
    #             f"Request info: {request_info}"
    #         )
    #         logger.error(error_msg)
    #         return self._generate_fallback_response()
            
    #     except Exception as api_error:
    #         error_type = type(api_error).__name__
    #         error_msg = (
    #             f"API call error: {error_type}\n"
    #             f"Error message: {str(api_error)}\n"
    #             f"Request info: {request_info}\n"
    #             f"API endpoint: {model_config.api_endpoint}\n"
    #             f"Model: {model_config.name}\n"
    #             f"API key: {'SET' if model_config.api_key else 'NOT SET'}\n"
    #             f"Base URL: {base_url if 'base_url' in locals() else 'N/A'}"
    #         )
            
    #         # 如果是 OpenAI API 特定错误，添加更多信息
    #         if hasattr(api_error, 'response'):
    #             error_msg += f"\nResponse status: {getattr(api_error.response, 'status_code', 'N/A')}"
    #             error_msg += f"\nResponse body: {getattr(api_error.response, 'text', 'N/A')[:500]}"
            
    #         logger.error(error_msg)
    #         import traceback
    #         logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    #         return self._generate_fallback_response()


    async def _call_api_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        image_data: Optional[List[str]] = None
    ) -> Dict:
        """调用API模型（参考 api_client.py 的实现方式）"""
        # 记录请求信息（用于错误诊断）
        request_info = {
            "model": model_config.name,
            "endpoint": model_config.api_endpoint,
            "prompt_length": len(prompt),
            "image_count": len(image_data) if image_data else 0,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature
        }
        
        try:
            from openai import AsyncOpenAI
            from PIL import Image
            from io import BytesIO
            import base64
            import asyncio
            
            logger.info(f"Calling API model {model_config.name} for evaluation...")
            logger.debug(f"API endpoint: {model_config.api_endpoint}")
            logger.debug(f"Prompt length: {len(prompt)} chars, Images: {len(image_data) if image_data else 0}")
            logger.debug(f"Request parameters: max_tokens={model_config.max_tokens}, temperature={model_config.temperature}")
            
            # 构建消息内容（参考 api_client.py 的 _build_messages 方法）
            content = []
            
            if image_data:
                # 处理图片：将 base64 字符串转换为 Image 对象，然后重新编码（与 api_client.py 保持一致）
                for img_b64 in image_data:
                    try:
                        # 解码 base64 字符串
                        img_bytes = base64.b64decode(img_b64)
                        # 转换为 PIL Image
                        img = Image.open(BytesIO(img_bytes))
                        # 确保是 RGB 模式
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 重新编码为 base64（与 api_client.py 保持一致）
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG", quality=85)
                        img_b64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64_encoded}"}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process image: {e}")
                        # 如果处理失败，直接使用原始 base64
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
            
            # 添加文本提示
            content.append({"type": "text", "text": prompt})
            
            # 构建消息（与 api_client.py 保持一致）
            messages = [{"role": "user", "content": content}]
            
            # 验证 API key 是否存在
            if not model_config.api_key or model_config.api_key == "NOT SET" or len(model_config.api_key.strip()) == 0:
                error_msg = (
                    f"API key is not set for model {model_config.name}\n"
                    f"Request info: {request_info}\n"
                    f"Please check config.json and ensure api_key is set"
                )
                logger.error(error_msg)
                return self._generate_fallback_response()
            
            # 按照正确示例创建客户端
            # config中的api_endpoint就是base_url，直接使用
            base_url = model_config.api_endpoint.rstrip('/') if model_config.api_endpoint else None
            if not base_url:
                logger.error(f"API endpoint is not set for model {model_config.name}")
                return self._generate_fallback_response()
            
            # 如果base_url不以/v1结尾，自动添加（兼容旧配置）
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
                logger.debug(f"自动添加 /v1 路径，使用 base_url: {base_url}")
            
            logger.debug(f"Using base_url: {base_url}")
            
            # 创建OpenAI异步客户端（添加超时设置）
            client = AsyncOpenAI(
                api_key=model_config.api_key,
                base_url=base_url,
                timeout=120.0,
                max_retries=2
            )
            
            logger.info(f"Sending request to {model_config.name}...")
            
            # 调用API（添加超时保护）
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_config.name,
                        messages=messages,
                        max_tokens=model_config.max_tokens,
                        temperature=model_config.temperature
                    ),
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                error_msg = f"API request timed out after 300(yaun 150) seconds for model {model_config.name}"
                logger.error(error_msg)
                return self._generate_fallback_response()
            
            result_text = response.choices[0].message.content
            logger.info(f"API call successful, response length: {len(result_text)} chars")
            
            # 记录token使用情况（参考 api_client.py）
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Token usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
            
            # 优化的JSON解析逻辑
            import re
            
            # 清理响应文本：移除markdown代码块标记
            cleaned_text = result_text.strip()
            
            # 移除 ```json 或 ``` 包裹
            if cleaned_text.startswith('```'):
                # 移除开头的 ```json 或 ```
                cleaned_text = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_text)
                # 移除结尾的 ```
                cleaned_text = re.sub(r'\n?```\s*$', '', cleaned_text)
                cleaned_text = cleaned_text.strip()
            
            # 尝试多种解析策略
            result_dict = None
            
            # 策略1: 直接解析清理后的文本
            try:
                result_dict = json.loads(cleaned_text)
                logger.debug("Successfully parsed JSON response (direct)")
            except json.JSONDecodeError:
                pass
            
            # 策略2: 提取第一个完整的JSON对象
            if result_dict is None:
                json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        result_dict = json.loads(json_match.group())
                        logger.debug("Successfully parsed JSON response (regex extraction)")
                    except json.JSONDecodeError:
                        pass
            
            # 策略3: 查找最大的JSON对象（处理嵌套情况）
            if result_dict is None:
                json_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                for match in json_matches:
                    try:
                        result_dict = json.loads(match.group())
                        logger.debug("Successfully parsed JSON response (nested extraction)")
                        break
                    except json.JSONDecodeError:
                        continue
            
            # 如果所有策略都失败
            if result_dict is None:
                logger.error("Failed to parse response as JSON")
                logger.error(f"Original response (first 500 chars): {result_text[:500]}")
                logger.error(f"Cleaned response (first 500 chars): {cleaned_text[:500]}")
                return self._generate_fallback_response()
            
            return result_dict
            
        except ImportError as e:
            error_msg = (
                f"Missing required library: {e}\n"
                f"Please install: pip install openai\n"
                f"Request info: {request_info}"
            )
            logger.error(error_msg)
            return self._generate_fallback_response()
            
        except Exception as api_error:
            error_type = type(api_error).__name__
            error_msg = (
                f"API call error: {error_type}\n"
                f"Error message: {str(api_error)}\n"
                f"Request info: {request_info}\n"
                f"API endpoint: {model_config.api_endpoint}\n"
                f"Model: {model_config.name}\n"
                f"API key: {'SET' if model_config.api_key else 'NOT SET'}\n"
                f"Base URL: {base_url if 'base_url' in locals() else 'N/A'}"
            )
            
            # 如果是 OpenAI API 特定错误，添加更多信息
            if hasattr(api_error, 'response'):
                error_msg += f"\nResponse status: {getattr(api_error.response, 'status_code', 'N/A')}"
                error_msg += f"\nResponse body: {getattr(api_error.response, 'text', 'N/A')[:500]}"
            
            logger.error(error_msg)
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return self._generate_fallback_response()
    
    async def _call_local_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        image_data: Optional[List[str]] = None
    ) -> Dict:
        """调用本地模型"""
        try:
            logger.info(f"Calling local model {model_config.name} for evaluation...")
            
            # 加载模型（如果尚未加载）
            model_data = self._load_local_model(model_config)
            
            if model_data is None:
                logger.error("Local model not loaded, cannot proceed")
                return self._generate_fallback_response()
            
            from PIL import Image
            import io
            import base64
            
            # 检查是否使用 QwenVLClientVLLM
            use_mllm_client = model_data.get("use_mllm_client", False)
            
            if use_mllm_client and "client" in model_data:
                # 使用 QwenVLClientVLLM 客户端
                client = model_data["client"]
                
                # 处理图片：将 base64 转换为 PIL Image
                images = []
                if image_data:
                    for img_b64 in image_data:
                        try:
                            img_bytes = base64.b64decode(img_b64)
                            image = Image.open(io.BytesIO(img_bytes))
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            images.append(image)
                        except Exception as e:
                            logger.warning(f"Failed to decode image: {e}")
                
                # 使用线程池执行同步推理（避免阻塞事件循环）
                import asyncio
                loop = asyncio.get_event_loop()
                
                def run_inference():
                    """在同步函数中运行推理"""
                    # 调用客户端的 chat 方法
                    result = client.chat(
                        prompt=prompt,
                        images=images,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens
                    )
                    # client.chat 返回 {"text": str, "json": Optional[Dict]}
                    return result.get("text", "")
                
                # 在线程池中运行推理（避免阻塞）
                result_text = await loop.run_in_executor(None, run_inference)
                logger.info(f"Local model inference completed, response length: {len(result_text)} chars")
                
                # 解析JSON响应
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        result_dict = json.loads(json_match.group())
                        logger.debug("Successfully parsed JSON response from local model")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from local model response: {e}")
                        logger.warning(f"Response text (first 500 chars): {result_text[:500]}")
                        # 尝试解析整个响应
                        try:
                            result_dict = json.loads(result_text)
                        except:
                            logger.error("Failed to parse response as JSON, using fallback")
                            return self._generate_fallback_response()
                else:
                    # 尝试直接解析整个响应
                    try:
                        result_dict = json.loads(result_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Response is not valid JSON: {e}")
                        logger.error(f"Response text (first 500 chars): {result_text[:500]}")
                        return self._generate_fallback_response()
                
                return result_dict
            
            # 原有的 transformers/vllm 调用方式（保留兼容性）
            import torch
            model = model_data.get("model")
            device = model_data.get("device", "cpu")
            model_type = model_data.get("model_type", "text")
            
            # 使用线程池执行同步推理（避免阻塞事件循环）
            import asyncio
            loop = asyncio.get_event_loop()
            
            def run_inference():
                """在同步函数中运行推理（参考 mllm_client.py 的实现）"""
                use_vllm = model_data.get("use_vllm", False)
                
                if use_vllm and "llm" in model_data:
                    # 使用 vllm 进行推理（参考 mllm_client.py 的 chat 方法）
                    llm = model_data["llm"]
                    SamplingParams = model_data["SamplingParams"]
                    
                    # 处理图片：将 base64 转换为 PIL Image（参考 mllm_client.py 的 _build_messages）
                    images = []
                    if image_data:
                        for img_b64 in image_data:
                            try:
                                img_bytes = base64.b64decode(img_b64)
                                image = Image.open(io.BytesIO(img_bytes))
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                images.append(image)
                            except Exception as e:
                                logger.warning(f"Failed to decode image: {e}")
                    
                    # 构建消息格式（参考 mllm_client.py 的 _build_messages）
                    content = []
                    
                    # 添加图片（使用 image_url 格式，参考 mllm_client.py）
                    for img in images:
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")  # 使用 PNG 格式
                        img_bytes = buffered.getvalue()
                        img_b64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                        data_url = f"data:image/png;base64,{img_b64_encoded}"
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                    
                    # 添加文本
                    content.append({
                        "type": "text",
                        "text": prompt
                    })
                    
                    # 构建消息（参考 mllm_client.py）
                    messages = [{
                        "role": "user",
                        "content": content
                    }]
                    
                    # 设置采样参数（参考 mllm_client.py）
                    sampling_params = SamplingParams(
                        temperature=model_config.temperature,
                        top_p=0.95 if model_config.temperature > 0 else 1.0,
                        max_tokens=model_config.max_tokens
                    )
                    
                    # 使用 llm.chat() 方法（参考 mllm_client.py）
                    outputs = llm.chat(
                        messages=messages,
                        sampling_params=sampling_params
                    )
                    
                    # 提取生成的文本（参考 mllm_client.py）
                    if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
                        logger.error("No output from vLLM")
                        result_text = ""
                    else:
                        result_text = outputs[0].outputs[0].text.strip()
                    
                elif model_type == "vl" and "processor" in model_data:
                    # Vision-Language模型（使用 transformers）
                    # 使用 torch.inference_mode() 替代 no_grad()，更节省内存
                    with torch.inference_mode():
                        processor = model_data["processor"]
                        model = model_data["model"]
                        
                        # 处理图片
                        images = []
                        if image_data:
                            for img_b64 in image_data:
                                try:
                                    img_bytes = base64.b64decode(img_b64)
                                    image = Image.open(io.BytesIO(img_bytes))
                                    if image.mode != 'RGB':
                                        image = image.convert('RGB')
                                    images.append(image)
                                except Exception as e:
                                    logger.warning(f"Failed to decode image: {e}")
                        
                        # 构建输入（Qwen3-VL 格式）
                        if images:
                            # 多图支持：构建消息格式
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        *[{"type": "image", "image": img} for img in images],
                                        {"type": "text", "text": prompt}
                                    ]
                                }
                            ]
                            
                            # 使用 apply_chat_template 处理消息
                            text = processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            
                            # Qwen3VLProcessor 可以直接处理图片列表和文本
                            # 不需要 process_vision_info 方法
                            try:
                                # 方法1：直接传入图片列表和文本
                                inputs = processor(
                                    text=[text],
                                    images=images,
                                    padding=True,
                                    return_tensors="pt"
                                )
                            except Exception as e:
                                # 如果上面的方法失败，尝试另一种方式
                                logger.warning(f"Direct image processing failed: {e}, trying alternative method")
                                # 方法2：使用 messages 格式（某些版本的 processor 支持）
                                try:
                                    inputs = processor(
                                        messages=messages,
                                        padding=True,
                                        return_tensors="pt"
                                    )
                                except Exception as e2:
                                    logger.error(f"Alternative method also failed: {e2}")
                                    # 方法3：最简单的文本+图片方式
                                    inputs = processor(
                                        text=prompt,
                                        images=images if len(images) == 1 else images[:1],  # 如果多图失败，只取第一张
                                        padding=True,
                                        return_tensors="pt"
                                    )
                        else:
                            # 无图片，仅文本
                            inputs = processor(
                                text=prompt,
                                padding=True,
                                return_tensors="pt"
                            )
                        
                        # 移动到设备
                        if device != "cpu":
                            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        else:
                            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        
                        # 生成
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=model_config.max_tokens,
                            temperature=model_config.temperature,
                            do_sample=True if model_config.temperature > 0 else False
                        )
                        
                        # 解码（跳过输入部分）
                        # inputs 是字典，需要使用 inputs["input_ids"] 而不是 inputs.input_ids
                        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                        ]
                        result_text = processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                
                elif model_type == "text" and "tokenizer" in model_data:
                    # 纯语言模型
                    # 使用 torch.inference_mode() 替代 no_grad()，更节省内存
                    with torch.inference_mode():
                        tokenizer = model_data["tokenizer"]
                        model = model_data["model"]
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                        
                        # 移动到设备
                        if device != "cpu":
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        else:
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # 生成
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=model_config.max_tokens,
                            temperature=model_config.temperature,
                            do_sample=True if model_config.temperature > 0 else False
                        )
                        
                        # 解码（跳过输入部分）
                        # inputs 是字典，需要使用 inputs["input_ids"] 而不是 inputs.input_ids
                        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                        ]
                        result_text = tokenizer.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                else:
                    raise ValueError(f"Unknown model type or missing processor/tokenizer: {model_type}")
                
                return result_text
            
            # 在线程池中运行推理（避免阻塞）
            result_text = await loop.run_in_executor(None, run_inference)
            logger.info(f"Local model inference completed, response length: {len(result_text)} chars")
            
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result_dict = json.loads(json_match.group())
                    logger.debug("Successfully parsed JSON response from local model")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from local model response: {e}")
                    logger.warning(f"Response text (first 500 chars): {result_text[:500]}")
                    # 尝试解析整个响应
                    try:
                        result_dict = json.loads(result_text)
                    except:
                        logger.error("Failed to parse response as JSON, using fallback")
                        return self._generate_fallback_response()
            else:
                # 尝试直接解析整个响应
                try:
                    result_dict = json.loads(result_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Response is not valid JSON: {e}")
                    logger.error(f"Response text (first 500 chars): {result_text[:500]}")
                    return self._generate_fallback_response()
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Local model call failed: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._generate_fallback_response()
    
    async def _call_huggingface_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        image_data: Optional[List[str]] = None
    ) -> Dict:
        """调用HuggingFace模型"""
        try:
            logger.info(f"Calling HuggingFace model {model_config.hf_model_id} for evaluation...")
            
            # 加载模型（如果尚未加载）
            model_data = self._load_huggingface_model(model_config)
            
            if model_data is None:
                # 如果模型加载失败，使用模拟响应
                logger.warning("HuggingFace model not loaded, using mock response")
                result_text = self._generate_mock_response(model_config.capability)
            else:
                # 实际调用HuggingFace模型
                """
                import torch
                from PIL import Image
                import io
                import base64
                
                model = model_data["model"]
                
                # 处理输入
                if image_data and "processor" in model_data:
                    # Vision-Language模型
                    processor = model_data["processor"]
                    first_image = image_data[0] if isinstance(image_data, list) else image_data
                    image = Image.open(io.BytesIO(base64.b64decode(first_image)))
                    inputs = processor(text=prompt, images=image, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=model_config.max_tokens, temperature=model_config.temperature)
                    result_text = processor.decode(outputs[0], skip_special_tokens=True)
                elif "tokenizer" in model_data:
                    # 纯语言模型
                    tokenizer = model_data["tokenizer"]
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=model_config.max_tokens, temperature=model_config.temperature)
                    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    result_text = self._generate_mock_response(model_config.capability)
                """
                # 暂时使用模拟响应
                result_text = self._generate_mock_response(model_config.capability)
            
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                result_dict = json.loads(result_text)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"HuggingFace model call failed: {e}")
            return self._generate_fallback_response()
    
    def _generate_mock_response(self, capability: str) -> str:
        """生成模拟响应(用于演示)"""
        if capability == "weak":
            return json.dumps({
                "correctness_score": 7.5,
                "correctness_confidence": 0.7,
                "correctness_rationale": "数据基本准确,但细节有待确认",
                "training_quality_score": 7.0,
                "training_quality_confidence": 0.6,
                "training_quality_rationale": "样本质量中等,可用于训练",
                "recommended_metrics": [
                    {
                        "general_metric_description": "多模态对话数据的清晰度和可理解性",
                        "specific_metric_description": "当前对话数据中，question和answer是否清晰可辨，文本内容是否完整且易于理解，图片信息是否能够被准确描述"
                    },
                    {
                        "general_metric_description": "对话内容与目标任务的相关性",
                        "specific_metric_description": "当前对话数据中的question和answer是否与视觉理解任务相关，是否有助于训练模型理解图片内容并生成准确回答"
                    }
                ],
                "overall_assessment": "该样本质量合格,可以考虑用于训练"
            })
        
        elif capability == "normal":
            return json.dumps({
                "correctness_score": 8.2,
                "correctness_confidence": 0.82,
                "correctness_rationale": "数据准确性高,图文一致性良好,逻辑合理",
                "training_quality_score": 8.5,
                "training_quality_confidence": 0.85,
                "training_quality_rationale": "样本具有良好的代表性和信息量,对训练有积极作用",
                "recommended_metrics": [
                    {
                        "general_metric_description": "多模态对话中视觉与语言信息的语义一致性",
                        "specific_metric_description": "当前对话数据中，answer对图片内容的描述是否与question询问的内容在语义上高度一致，是否存在图文不匹配或语义偏差的情况"
                    },
                    {
                        "general_metric_description": "对话数据的信息密度和丰富度",
                        "specific_metric_description": "当前对话数据中，question是否包含足够的信息量来引导模型关注图片的关键内容，answer是否提供了丰富且详细的视觉信息描述"
                    },
                    {
                        "general_metric_description": "对话样本对模型训练的挑战性和学习价值",
                        "specific_metric_description": "当前对话数据的难度是否适中，question是否能够有效测试模型的视觉理解能力，answer是否包含有助于模型学习的关键信息"
                    }
                ],
                "overall_assessment": "高质量样本,建议纳入训练集"
            })
        
        else:  # SOTA
            return json.dumps({
                "correctness_score": 8.7,
                "correctness_confidence": 0.92,
                "correctness_rationale": "数据准确性优秀,多模态一致性强,语义完整,逻辑严密,未发现明显偏差",
                "training_quality_score": 9.0,
                "training_quality_confidence": 0.90,
                "training_quality_rationale": "优质训练样本,具有高代表性、丰富信息密度和适当挑战性,预期对模型性能提升有显著贡献",
                "recommended_metrics": [
                    {
                        "general_metric_description": "视觉-语言跨模态对齐的语义一致性质量",
                        "specific_metric_description": "当前对话数据中，answer对图片的视觉描述是否与question的询问意图完美对齐，是否存在跨模态语义不一致、描述偏差或理解错误的情况，评分时检查视觉信息与语言表达的匹配度"
                    },
                    {
                        "general_metric_description": "对话数据的信息熵和语义复杂度",
                        "specific_metric_description": "当前对话数据中，question和answer是否包含丰富的语义信息和结构信息，是否涉及复杂的视觉理解任务(如空间关系、物体属性、场景理解等)，信息密度是否足够高以提供有效的训练信号"
                    },
                    {
                        "general_metric_description": "对话样本对数据集多样性和覆盖度的贡献",
                        "specific_metric_description": "当前对话数据是否包含独特的问题类型、视觉场景或回答模式，是否能够补充现有数据集中缺失的视觉理解维度，是否有助于提升模型在不同场景下的泛化能力"
                    },
                    {
                        "general_metric_description": "对话样本产生有效学习梯度和训练信号的质量",
                        "specific_metric_description": "当前对话数据的难度是否适中(既不过于简单也不过于困难)，question是否能够有效引导模型学习关键的视觉理解能力，answer是否提供了清晰、准确的学习目标，是否能够产生高质量的反向传播梯度"
                    },
                    {
                        "general_metric_description": "对话样本用于测试和提升模型鲁棒性的价值",
                        "specific_metric_description": "当前对话数据是否包含边界情况、罕见场景或具有挑战性的视觉理解任务，是否能够测试模型在复杂情况下的表现，是否有助于提升模型的抗干扰能力和泛化性能"
                    }
                ],
                "overall_assessment": "顶级训练样本,强烈推荐纳入训练集。该样本在多个维度上表现优异,预期可显著提升模型在相关任务上的性能。无明显风险或负面影响。"
            })
    
    def _generate_fallback_response(self) -> Dict:
        """生成回退响应"""
        return {
            "correctness_score": 5.0,
            "correctness_confidence": 0.5,
            "correctness_rationale": "评估失败,使用默认值",
            "training_quality_score": 5.0,
            "training_quality_confidence": 0.5,
            "training_quality_rationale": "评估失败,使用默认值",
            "recommended_metrics": [],
            "overall_assessment": "评估过程出现错误"
        }
    
    def _prepare_image_data(self, sample: DataSample) -> List[str]:
        """准备图片数据，支持多图"""
        try:
            if not sample.image_base64:
                return []
            # 如果是单个字符串则包装成列表，兼容老格式
            if isinstance(sample.image_base64, str):
                return [sample.image_base64]
            # 已是列表
            return list(sample.image_base64)
        except Exception as e:
            logger.error(f"图片数据准备失败: {e}")
            return []
    
    async def evaluate_sample(
        self,
        sample: DataSample,
        model_capability: ModelCapability,
        use_simple_evaluation: bool = False,
        test_mode: bool = False
    ) -> EvaluationResult:
        """
        使用指定能力级别的模型评估单个样本
        
        参数:
            sample: 数据样本
            model_capability: 模型能力级别
            use_simple_evaluation: 是否使用简单版评估（True=简单版，False=复杂版）
            
        返回:
            EvaluationResult: 评估结果
        """
        mode_str = f"{'简单版' if use_simple_evaluation else '复杂版'}{' + 测试模式' if test_mode else ''}"
        logger.info(f"使用 {model_capability.value} 模型评估样本 {sample.sample_id} (评估模式: {mode_str})")
        
        model_config = self.models[model_capability]
        prompt = self._create_evaluation_prompt(sample, model_capability, use_simple_evaluation, test_mode)
        image_data = self._prepare_image_data(sample)
        
        # 调用模型API
        result_dict = await self._call_model(model_config, prompt, image_data)
        
        # 构建评估结果
        from datetime import datetime
        
        # 处理recommended_metrics
        # 转换为字典列表格式，每个字典包含 general_metric_description 和 specific_metric_description
        metrics = []
        for m in result_dict.get('recommended_metrics', []):
            metric_dict = {
                "general_metric_description": m.get('general_metric_description', ''),
                "specific_metric_description": m.get('specific_metric_description', '')
            }
            metrics.append(metric_dict)
        
        evaluation_result = EvaluationResult(
            model_name=model_config.name,
            model_capability=model_capability,
            sample_id=sample.sample_id,
            correctness_score=result_dict['correctness_score'],
            correctness_confidence=result_dict['correctness_confidence'],
            correctness_rationale=result_dict['correctness_rationale'],
            training_quality_score=result_dict['training_quality_score'],
            training_quality_confidence=result_dict['training_quality_confidence'],
            training_quality_rationale=result_dict['training_quality_rationale'],
            recommended_metrics=metrics,
            overall_assessment=result_dict['overall_assessment'],
            timestamp=datetime.now().isoformat(),
            information_recap=result_dict.get('information_recap')  # 测试模式的复述信息
        )
        
        self.evaluation_history.append(evaluation_result)
        return evaluation_result
    
    async def evaluate_sample_all_models(
        self,
        sample: DataSample,
        use_simple_evaluation: bool = False,
        test_mode: bool = False
    ) -> Dict[ModelCapability, EvaluationResult]:
        """
        使用所有三个级别的模型评估单个样本
        
        参数:
            sample: 数据样本
            use_simple_evaluation: 是否使用简单版评估（True=简单版，False=复杂版）
            
        返回:
            Dict[ModelCapability, EvaluationResult]: 各模型的评估结果
        """
        mode_str = f"{'简单版' if use_simple_evaluation else '复杂版'}{' + 测试模式' if test_mode else ''}"
        logger.info(f"使用三级模型评估样本 {sample.sample_id} (评估模式: {mode_str})")
        
        results = {}
        for capability in ModelCapability:
            result = await self.evaluate_sample(sample, capability, use_simple_evaluation, test_mode)
            results[capability] = result
        
        return results
    
    async def batch_evaluate(
        self,
        samples: List[DataSample],
        use_all_models: bool = True,
        use_simple_evaluation: bool = False,
        test_mode: bool = False
    ) -> List[Dict[ModelCapability, EvaluationResult]]:
        """
        批量评估多个样本
        
        参数:
            samples: 数据样本列表
            use_all_models: 是否使用所有模型
            use_simple_evaluation: 是否使用简单版评估（True=简单版，False=复杂版）
            test_mode: 是否使用测试模式（True=仅处理前3个样本，并要求模型复述信息）
            
        返回:
            List[Dict[ModelCapability, EvaluationResult]]: 评估结果列表
        """
        # 测试模式：仅处理前3个样本
        if test_mode:
            samples = samples[:3]
            logger.info(f"测试模式：仅处理前3个样本")
        
        mode_str = f"{'简单版' if use_simple_evaluation else '复杂版'}{' + 测试模式' if test_mode else ''}"
        logger.info(f"开始批量评估 {len(samples)} 个样本 (评估模式: {mode_str})")
        
        results = []
        for sample in samples:
            if use_all_models:
                sample_results = await self.evaluate_sample_all_models(sample, use_simple_evaluation, test_mode)
            else:
                # 只使用SOTA模型
                sample_results = {
                    ModelCapability.SOTA: await self.evaluate_sample(
                        sample, ModelCapability.SOTA, use_simple_evaluation, test_mode
                    )
                }
            results.append(sample_results)
        
        logger.info(f"批量评估完成,共评估 {len(results)} 个样本")
        return results
    
    def save_results(
        self,
        results: List[Dict[ModelCapability, EvaluationResult]],
        output_path: str
    ):
        """
        保存评估结果
        
        参数:
            results: 评估结果列表
            output_path: 输出文件路径
        """
        logger.info(f"保存评估结果到 {output_path}")
        
        # 转换为可序列化的格式
        serializable_results = []
        for sample_results in results:
            sample_dict = {}
            for capability, result in sample_results.items():
                result_dict = asdict(result)
                result_dict['model_capability'] = capability.value
                sample_dict[capability.value] = result_dict
            serializable_results.append(sample_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info("评估结果保存完成")


def load_samples_from_file(sampled_file_path: str) -> List[DataSample]:
    """
    从采样结果文件中加载DataSample列表
    
    参数:
        sampled_file_path: 采样结果JSON文件路径
        
    返回:
        List[DataSample]: DataSample列表
    """
    logger.info(f"从文件加载样本: {sampled_file_path}")
    
    with open(sampled_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples_data = data.get('samples', [])
    cluster_id = data.get('cluster_id', 'unknown')
    
    logger.info(f"Cluster {cluster_id}: 加载了 {len(samples_data)} 个样本")
    
    data_samples = []
    for sample_dict in samples_data:
        try:
            data_sample = DataSample(
                sample_id=str(sample_dict.get('sample_id', '')),
                image_base64=sample_dict.get('image_base64'),
                dialogue=sample_dict.get('dialogue'),
                metadata=sample_dict.get('metadata')
            )
            data_samples.append(data_sample)
        except Exception as e:
            logger.error(f"加载样本失败 (sample_id={sample_dict.get('sample_id', 'unknown')}): {e}")
    
    return data_samples


def load_samples_from_directory(sampled_dir: str, pattern: str = "cluster_*_samples.json") -> List[DataSample]:
    """
    从目录中加载所有采样结果文件
    
    参数:
        sampled_dir: 采样结果目录
        pattern: 文件匹配模式
        
    返回:
        List[DataSample]: 所有DataSample列表
    """
    sampled_path = Path(sampled_dir)
    all_samples = []
    
    # 查找所有匹配的文件
    sample_files = list(sampled_path.glob(pattern))
    logger.info(f"在目录 {sampled_dir} 中找到 {len(sample_files)} 个采样结果文件")
    
    for sample_file in sorted(sample_files):
        samples = load_samples_from_file(str(sample_file))
        all_samples.extend(samples)
    
    return all_samples


# 使用示例
async def main():
    """主函数 - 支持从采样结果文件读取"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据质量评估系统')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--input_file', type=str, default=None,
                       help='单个采样结果JSON文件路径')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='采样结果目录（包含多个cluster_*_samples.json文件）')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='评估结果输出文件路径')
    parser.add_argument('--use_all_models', action='store_true', default=True,
                       help='使用所有三个模型进行评估')
    parser.add_argument('--use_simple_evaluation', action='store_true', default=False,
                       help='使用简单版评估')
    parser.add_argument('--test_mode', action='store_true', default=False,
                       help='测试模式（仅处理前3个样本，并要求模型复述信息）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("三级模型数据质量评估系统")
    print("=" * 80)
    
    # 创建评估器
    evaluator = DataQualityEvaluator(config_path=args.config)
    
    # 加载样本数据
    samples = []
    
    if args.input_file:
        # 从单个文件加载
        samples = load_samples_from_file(args.input_file)
    elif args.input_dir:
        # 从目录加载所有文件
        samples = load_samples_from_directory(args.input_dir)
    else:
        # 使用默认示例数据
        print("未指定输入文件，使用示例数据")
        samples = [
            DataSample(
                sample_id="sample_001",
                dialogue=[
                    {"question": "图片中有什么动物？", "answer": "一只可爱的橘猫"},
                    {"question": "它在做什么？", "answer": "坐在窗台上,看着外面的鸟儿"}
                ],
                image_base64=None,
                metadata={"source": "dataset_A", "category": "animal"}
            ),
            DataSample(
                sample_id="sample_002",
                dialogue=[
                    {"question": "这是什么场景？", "answer": "现代化的办公室"},
                    {"question": "人们在做什么？", "answer": "员工们正在使用笔记本电脑工作"}
                ],
                image_base64=None,
                metadata={"source": "dataset_B", "category": "workplace"}
            )
        ]
    
    if not samples:
        print("错误: 没有找到可评估的样本")
        return
    
    print(f"\n共有 {len(samples)} 个样本待评估")
    
    # 批量评估
    mode_str = f"{'简单版' if args.use_simple_evaluation else '复杂版'}{' + 测试模式' if args.test_mode else ''}"
    print(f"\n评估模式: {mode_str}")
    print("开始批量评估...\n")
    
    results = await evaluator.batch_evaluate(
        samples,
        use_all_models=args.use_all_models,
        use_simple_evaluation=args.use_simple_evaluation,
        test_mode=args.test_mode
    )
    
    # 显示结果
    for i, sample_results in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"样本 {i}: {samples[i-1].sample_id}")
        print(f"{'='*80}")
        
        for capability, result in sample_results.items():
            print(f"\n【{capability.value.upper()} 模型评估】")
            print(f"模型: {result.model_name}")
            
            # 测试模式：显示信息复述
            if result.information_recap:
                print(f"\n【信息复述（测试模式）】")
                print(f"{result.information_recap}")
            
            print(f"正确性评分: {result.correctness_score:.2f} (置信度: {result.correctness_confidence:.2f})")
            print(f"训练质量评分: {result.training_quality_score:.2f} (置信度: {result.training_quality_confidence:.2f})")
            print(f"\n推荐的筛选指标:")
            for j, metric in enumerate(result.recommended_metrics, 1):
                print(f"  {j}. 概括性标准: {metric['general_metric_description']}")
                print(f"     具体标准: {metric['specific_metric_description']}")
            print(f"\n综合评估: {result.overall_assessment}")
    
    # 保存结果
    evaluator.save_results(results, args.output)
    print(f"\n\n评估结果已保存到: {args.output}")
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())


# 单个采样文件评估 - 普通模式
# python -m extractor.evaluator \
#   --input_file /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/c_sampled_data/cluster_0_samples.json \
#   --output /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/d_extracted_sample_data/extra_evaluation.json \
#   --config config.json \
#   --test_mode

# # 整个目录评估 - 普通模式
# python evaluator.py \
#   --input_dir /user/zhuxuzhou/a_whole_pipeline/sampler/src/sampled_results \
#   --output evaluation_results_all.json \
#   --config config.json \
#   --use_all_models \
#   --test_mode

# # 正式模式评估
# python evaluator.py \
#   --input_dir /user/zhuxuzhou/a_whole_pipeline/sampler/src/sampled_results \
#   --output evaluation_results.json \
#   --config config.json
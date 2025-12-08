"""
MLLM 客户端模块
支持多种 MLLM 模型的统一接口，包括本地 QwenVL 模型

依赖安装:
    pip install transformers torch pillow
    # QwenVL 可能还需要: pip install qwen-vl-utils
"""

# 必须在导入任何可能使用 multiprocessing 的模块之前设置
import multiprocessing
import os

# 设置 multiprocessing start method 为 'spawn'，解决 vllm CUDA 初始化问题
# 必须在导入 vllm 之前设置
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 设置环境变量（双重保险）
os.environ.setdefault('VLLM_USE_MULTIPROCESSING_SPAWN', '1')
os.environ.setdefault('MULTIPROCESSING_METHOD', 'spawn')

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import json
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import base64
import io
from io import BytesIO
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseMLLMClient(ABC):
    """MLLM 客户端抽象基类"""
    
    @abstractmethod
    def chat(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        """
        发送多模态对话请求
        
        Args:
            prompt: 文本提示
            images: 图片列表
            **kwargs: 其他参数（如 temperature, max_tokens 等）
            
        Returns:
            模型返回的文本响应
        """
        pass
    
    @abstractmethod
    def chat_stream(self, prompt: str, images: List[Image.Image], **kwargs):
        """流式返回（可选实现）"""
        pass

class QwenVLClientVLLM(BaseMLLMClient):
    """
    QwenVL 本地模型客户端 (使用 vLLM)
    支持 Qwen2-VL 系列模型，提供高性能推理
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        初始化 QwenVL 客户端 (vLLM)
        
        Args:
            model_path: 本地模型路径
            tensor_parallel_size: 张量并行大小（多GPU）
            gpu_memory_utilization: GPU 内存利用率 (0.0-1.0)
            max_model_len: 最大模型长度（上下文窗口）
            max_num_seqs: 最大并发序列数
            temperature: 采样温度
            top_p: nucleus 采样参数
            max_tokens: 最大生成 token 数
            dtype: 数据类型 ("auto", "float16", "bfloat16")
            trust_remote_code: 是否信任远程代码
        """
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        logger.info(f"Loading QwenVL model with vLLM from {model_path}...")
        
        # 检查是否是 MoE 模型（MoE 模型 vllm 支持不好，容易出现 CUDA fork 错误）
        is_moe = "moe" in model_path.lower() or "A3B" in model_path or "moe" in str(model_path).lower()
        
        if is_moe:
            logger.warning(
                f"MoE model detected ({model_path}). "
                "vLLM has known issues with MoE models (CUDA fork errors). "
                "Attempting to load anyway, but if it fails, consider using transformers instead."
            )
        
        try:
            # 再次确保 multiprocessing start method 是 spawn（在导入 vllm 之前）
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            from vllm import LLM, SamplingParams
            from vllm.multimodal.utils import fetch_image
            
            self.SamplingParams = SamplingParams
            self.fetch_image = fetch_image
            
            # 检查环境变量，决定是否使用 CUDA 图
            # 如果设置了 VLLM_DISABLE_CUDA_GRAPH=1，则强制使用 eager 模式
            disable_cuda_graph = os.environ.get('VLLM_DISABLE_CUDA_GRAPH', '0') == '1'
            enforce_eager = disable_cuda_graph or os.environ.get('VLLM_ENFORCE_EAGER', '1') == '1'
            
            if disable_cuda_graph:
                logger.info("CUDA graphs disabled via environment variable VLLM_DISABLE_CUDA_GRAPH=1")
            
            # 初始化 vLLM 引擎
            # enforce_eager=True 会禁用 CUDA 图优化，节省内存但可能略微降低性能
            llm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
                "limit_mm_per_prompt": {"image": 10},  # 每个 prompt 最多 10 张图片
                "enforce_eager": enforce_eager,  # 强制使用 eager 模式（不使用 CUDA 图）
            }
            
            # 某些 vLLM 版本支持 use_cuda_graph 参数，如果不支持会忽略
            if not enforce_eager:
                llm_kwargs["use_cuda_graph"] = True
            
            self.llm = LLM(**llm_kwargs)
            
            logger.info("vLLM model loaded successfully!")
            
        except ImportError:
            raise ImportError(
                "vLLM not installed. Please install: pip install vllm"
            )
        except RuntimeError as e:
            if "Cannot re-initialize CUDA in forked subprocess" in str(e) or "spawn" in str(e).lower():
                logger.error(
                    f"vLLM CUDA multiprocessing error: {e}\n"
                    "This is a known issue with vLLM and MoE models. "
                    "The multiprocessing start method should be 'spawn', but vLLM may still use 'fork'. "
                    "For MoE models, consider using transformers instead."
                )
            raise
        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {e}")
            raise
    
    def chat(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        使用 vLLM 进行对话
        
        Args:
            prompt: 文本提示
            images: 图片列表
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数
            top_p: 覆盖默认 top_p
            
        Returns:
            模型生成的文本
        """
        # 使用传入的参数或默认值
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p
        
        # 构建消息格式
        messages = self._build_messages(prompt, images)
        
        # 设置采样参数
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        try:
            # 使用 vLLM 生成
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=sampling_params,
            )

            if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
                logger.error("No output from VLLM")
                return {"text": "", "json": None}
            
            # 提取生成的文本
            response_text = outputs[0].outputs[0].text.strip()

            json_obj = None
            response_text = response_text.strip()  # 去掉首尾空格

            if not response_text:
                logger.warning("Response is empty.")
                return {"text": response_text, "json": None}

            try:
                # 尝试匹配 ```json ... ``` 包裹
                match = re.search(r"```json\s*(\{.*)", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    # 截取到最后一个闭合大括号，避免截断导致解析失败
                    last_brace = json_str.rfind("}")
                    if last_brace != -1:
                        json_str = json_str[:last_brace+1]
                else:
                    json_str = response_text

                # 替换 Python 风格布尔/None
                json_str = json_str.replace("None", "null").replace("True", "true").replace("False", "false")

                # 尝试解析 JSON
                json_obj = json.loads(json_str)

            except json.JSONDecodeError as e:
                # 输出完整错误信息，便于调试
                logger.warning(
                    f"Failed to parse JSON from response.\n"
                    f"JSONDecodeError: {str(e)}\n"
                    f"Response snippet (first 500 chars):\n{response_text[:500]}"
                )

            return {"text": response_text, "json": json_obj}
        
        except Exception as e:
            logger.error(f"Error during vLLM inference: {e}")
            raise
    
    def _build_messages(
        self, 
        prompt: str, 
        images: List[Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        构建 vLLM 格式的消息
        
        Qwen2-VL 使用标准的 chat 格式：
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "..."}
                ]
            }
        ]
        """
        content = []

        # def image_to_base64(img_list):
        #     """
        #     将 PIL.Image 列表转换成 base64 字符串列表
        #     """
        #     base64_list = []
        #     for img in img_list:
        #         # 确保是 PIL.Image 对象
        #         if not isinstance(img, Image.Image):
        #             raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")
        #         buffered = io.BytesIO()
        #         img.save(buffered, format="PNG")  # 或 "JPEG"，根据模型支持
        #         img_bytes = buffered.getvalue()
        #         img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        #         base64_list.append(img_b64)
        #     return base64_list

        # images = image_to_base64(images)
        
        # 添加图片
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")
            # 转成 data URL
            buffered = BytesIO()
            img.save(buffered, format="PNG")  # 或 JPEG，根据模型支持
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{img_b64}"

            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        
        # 添加文本
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def chat_batch(
        self,
        prompts: List[str],
        images_list: List[List[Image.Image]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        批量处理（vLLM 的优势）
        
        Args:
            prompts: 文本提示列表
            images_list: 图片列表的列表
            temperature: 采样温度
            max_tokens: 最大 token 数
            top_p: nucleus 采样
            
        Returns:
            生成文本列表
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p
        
        # 构建批量消息
        messages_batch = [
            self._build_messages(prompt, images)
            for prompt, images in zip(prompts, images_list)
        ]
        
        # 设置采样参数
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        try:
            # 批量生成
            outputs = self.llm.chat(
                messages=messages_batch,
                sampling_params=sampling_params,
            )
            
            # 提取结果
            responses = []
            for output in outputs:
                if output.outputs:
                    responses.append(output.outputs[0].text.strip())
                else:
                    responses.append("")
            
            return responses
        
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            raise
    
    def chat_stream(self, prompt: str, images: List[Image.Image], **kwargs):
        """
        流式生成（vLLM 不原生支持流式，返回完整结果）
        
        注意：vLLM 0.4.0+ 版本支持流式输出，但 Qwen2-VL 可能不完全支持
        """
        response = self.chat(prompt, images, **kwargs)
        yield response



class QwenVLClient(BaseMLLMClient):
    """
    QwenVL 本地模型客户端
    支持 Qwen-VL 和 Qwen-VL-Chat 模型
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        """
        初始化 QwenVL 客户端
        
        Args:
            model_path: 本地模型路径或 HuggingFace 模型名称
            device: 设备 ("cuda" 或 "cpu")
            device_map: 设备映射策略
            trust_remote_code: 是否信任远程代码
            load_in_8bit: 是否使用 8bit 量化
            load_in_4bit: 是否使用 4bit 量化
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus 采样参数
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"Loading QwenVL model from {model_path}...")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # 加载模型
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def chat(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        使用 QwenVL 进行对话
        
        Args:
            prompt: 文本提示
            images: 图片列表
            temperature: 覆盖默认温度
            max_new_tokens: 覆盖默认最大 token 数
            top_p: 覆盖默认 top_p
            
        Returns:
            模型生成的文本
        """
        # 使用传入的参数或默认值
        temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        top_p = top_p if top_p is not None else self.top_p
        
        # 构建 QwenVL 格式的输入
        # QwenVL 使用特殊的格式: <img>image_path</img> 或直接传入图片
        query = self._build_query(prompt, images)
        
        try:
            # 方法1: 使用 chat 接口（推荐）
            if hasattr(self.model, 'chat'):
                response, history = self.model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                    images=images if images else None,
                )
                return response
            
            # 方法2: 使用 generate 接口
            else:
                inputs = self.tokenizer(query, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        **kwargs
                    )
                
                response = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # 移除输入部分，只保留生成的内容
                if query in response:
                    response = response.replace(query, "").strip()
                
                return response
        
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise
    
    def _build_query(self, prompt: str, images: List[Image.Image]) -> str:
        """
        构建 QwenVL 格式的查询
        
        QwenVL 可能需要特殊格式，具体取决于模型版本
        """
        # 如果没有图片，直接返回 prompt
        if not images:
            return prompt
        
        # 方式1: 直接使用 prompt（某些版本的 QwenVL 会自动处理图片）
        return prompt
        
        # 方式2: 如果需要特殊标记，可以使用类似下面的格式
        # image_tokens = "".join([f"<img>{i}</img>" for i in range(len(images))])
        # return f"{image_tokens}\n{prompt}"
    
    def chat_stream(self, prompt: str, images: List[Image.Image], **kwargs):
        """流式生成（QwenVL 可能不支持，返回完整结果）"""
        response = self.chat(prompt, images, **kwargs)
        yield response


class OpenAICompatibleClient(BaseMLLMClient):
    """
    OpenAI 兼容的 API 客户端
    支持 GPT-4V, Claude, 或其他兼容 OpenAI API 格式的服务
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """
        初始化 OpenAI 兼容客户端
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            max_tokens: 最大 token 数
            temperature: 采样温度
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def chat(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """使用 OpenAI 格式 API 进行对话"""
        import base64
        import io
        
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # 构建消息
        content = [{"type": "text", "text": prompt}]
        
        # 添加图片（转换为 base64）
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
        
        # 调用 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def chat_stream(self, prompt: str, images: List[Image.Image], **kwargs):
        """流式返回"""
        # 实现类似，使用 stream=True
        pass


class DummyMLLMClient(BaseMLLMClient):
    """
    虚拟客户端，用于测试
    返回固定的 JSON 格式响应
    """
    
    def __init__(self, always_pass: bool = True):
        """
        Args:
            always_pass: 如果为 True，总是返回通过的结果
        """
        self.always_pass = always_pass
    
    def chat(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        """返回虚拟的 JSON 响应"""
        if self.always_pass:
            response = {
                "linguistic_errors": [],
                "visual_inconsistencies": [],
                "has_error": False,
                "overall_severity": "low",
                "summary": "No issues found (dummy response)"
            }
        else:
            response = {
                "linguistic_errors": [
                    {
                        "type": "spelling",
                        "text": "teh",
                        "explanation": "Spelling error",
                        "suggestion": "the"
                    }
                ],
                "visual_inconsistencies": [],
                "has_error": True,
                "overall_severity": "medium",
                "summary": "Found spelling error (dummy response)"
            }
        
        return json.dumps(response, ensure_ascii=False)
    
    def chat_stream(self, prompt: str, images: List[Image.Image], **kwargs):
        """流式返回"""
        yield self.chat(prompt, images, **kwargs)


# ==================== 客户端工厂 ====================

class MLLMClientFactory:
    """MLLM 客户端工厂类"""
    
    @staticmethod
    def create_client(client_type: str, **kwargs) -> BaseMLLMClient:
        """
        创建 MLLM 客户端
        
        Args:
            client_type: 客户端类型 ("qwen-vl", "openai", "dummy")
            **kwargs: 传递给具体客户端的参数
            
        Returns:
            BaseMLLMClient 实例
        """
        if client_type.lower() in ["qwen-vl", "qwenvl", "qwen"]:
            return QwenVLClient(**kwargs)
        
        elif client_type.lower() in ["openai", "gpt", "claude"]:
            return OpenAICompatibleClient(**kwargs)
        
        elif client_type.lower() == "dummy":
            return DummyMLLMClient(**kwargs)
        
        else:
            raise ValueError(f"Unknown client type: {client_type}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例1: 使用本地 QwenVL 模型
    print("=== 示例1: 本地 QwenVL 模型 ===")
    
    # 初始化客户端
    client = QwenVLClient(
        model_path="/path/to/Qwen-VL-Chat",  # 替换为你的本地路径
        device="cuda",
        load_in_8bit=False,  # 如果显存不足，可以设为 True
        temperature=0.7,
        max_new_tokens=2048
    )
    
    # 准备测试数据
    test_prompt = """
Please check if there are any errors in the following answer:

Question: What color is the car in the image?
Answer: The car is blue with red stripes.

Output in JSON format with linguistic_errors and visual_inconsistencies.
"""
    
    test_images = [Image.new('RGB', (224, 224), color='blue')]
    
    # 调用模型
    try:
        response = client.chat(test_prompt, test_images)
        print("Model response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    
    # 示例2: 使用虚拟客户端测试
    print("\n=== 示例2: 虚拟客户端测试 ===")
    
    dummy_client = DummyMLLMClient(always_pass=True)
    response = dummy_client.chat(test_prompt, test_images)
    print("Dummy response:")
    print(response)
    
    # 示例3: 使用工厂创建
    print("\n=== 示例3: 使用工厂模式 ===")
    
    # 创建 QwenVL 客户端
    # client = MLLMClientFactory.create_client(
    #     "qwen-vl",
    #     model_path="/path/to/Qwen-VL-Chat",
    #     device="cuda"
    # )
    
    # 创建虚拟客户端
    client = MLLMClientFactory.create_client("dummy", always_pass=False)
    response = client.chat(test_prompt, test_images)
    print(json.dumps(json.loads(response), indent=2, ensure_ascii=False))
    
    # 示例4: 与过滤组件集成
    print("\n=== 示例4: 与过滤组件集成 ===")
    
    """
    from mllm_prompt_builder import MLLMPromptBuilder, ERROR_IN_ANSWER_TEMPLATE
    from vqa_data_structures import MultiImageVQAEntry
    
    # 创建客户端
    mllm_client = QwenVLClient(
        model_path="/path/to/Qwen-VL-Chat",
        device="cuda",
        temperature=0.1,  # 低温度更适合质检任务
    )
    
    # 创建过滤组件
    filter_component = MLLMFilterComponent(
        template=ERROR_IN_ANSWER_TEMPLATE,
        mllm_client=mllm_client,
        mode="single_turn"
    )
    
    # 过滤数据
    filtered_entry = filter_component.filter_entry(my_entry)
    """
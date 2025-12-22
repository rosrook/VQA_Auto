# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 模型适配器：统一接口，允许用户接入自己的模型
# """

# import abc
# from typing import Any, Dict, List, Optional, Union


# class BaseModelAdapter(abc.ABC):
#     """模型适配器基类"""
    
#     @abc.abstractmethod
#     def generate(self, 
#                 prompt: str,
#                 images: Optional[List[str]] = None,
#                 **kwargs) -> Dict[str, Any]:
#         """
#         生成模型响应
        
#         Args:
#             prompt: 文本提示
#             images: 图像路径列表或base64编码列表
#             **kwargs: 其他参数（temperature, max_tokens等）
        
#         Returns:
#             {
#                 "text": "模型输出文本",
#                 "usage": {...},  # 可选的token使用信息
#                 "raw": {...}     # 可选的原始响应
#             }
#         """
#         raise NotImplementedError
    
#     @abc.abstractmethod
#     def get_model_info(self) -> Dict[str, Any]:
#         """
#         获取模型信息
        
#         Returns:
#             {
#                 "name": "模型名称",
#                 "type": "模型类型",
#                 "version": "版本号"（可选）
#             }
#         """
#         raise NotImplementedError


# class ModelAdapterFactory:
#     """模型适配器工厂"""
    
#     _adapters = {}
    
#     @classmethod
#     def register(cls, name: str, adapter_class):
#         """注册适配器类"""
#         cls._adapters[name] = adapter_class
    
#     @classmethod
#     def create(cls, name: str, **kwargs) -> BaseModelAdapter:
#         """创建适配器实例"""
#         if name not in cls._adapters:
#             raise ValueError(f"未知的适配器类型: {name}")
#         return cls._adapters[name](**kwargs)
    
#     @classmethod
#     def list_adapters(cls) -> List[str]:
#         """列出所有已注册的适配器"""
#         return list(cls._adapters.keys())


# # 示例：OpenAI兼容的适配器
# class OpenAIAdapter(BaseModelAdapter):
#     """OpenAI兼容的API适配器"""
    
#     def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
#         self.api_key = api_key
#         self.base_url = base_url.rstrip("/")
#         self.model = model
#         self.kwargs = kwargs
    
#     def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
#         import requests
        
#         messages = [{"role": "user", "content": []}]
        
#         # 添加文本
#         messages[0]["content"].append({"type": "text", "text": prompt})
        
#         # 添加图像（如果提供）
#         if images:
#             for img in images:
#                 if img.startswith("http"):
#                     messages[0]["content"].append({"type": "image_url", "image_url": {"url": img}})
#                 else:
#                     # 假设是base64
#                     messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             **kwargs
#         }
        
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         response = requests.post(
#             f"{self.base_url}/chat/completions",
#             json=payload,
#             headers=headers,
#             timeout=kwargs.get("timeout", 30)
#         )
#         response.raise_for_status()
#         result = response.json()
        
#         return {
#             "text": result["choices"][0]["message"]["content"],
#             "usage": result.get("usage", {}),
#             "raw": result
#         }
    
#     def get_model_info(self) -> Dict[str, Any]:
#         return {
#             "name": self.model,
#             "type": "openai_compatible",
#             "base_url": self.base_url
#         }


# # HuggingFace本地模型适配器
# # class HuggingFaceAdapter(BaseModelAdapter):
# #     """HuggingFace本地模型适配器（支持文本和视觉-语言模型）"""
    
# #     def __init__(self, 
# #                  model_id: str,
# #                  device: str = "cuda",
# #                  dtype: str = "auto",
# #                  trust_remote_code: bool = False,
# #                  load_in_8bit: bool = False,
# #                  load_in_4bit: bool = False,
# #                  max_new_tokens: int = 1024,
# #                  temperature: float = 0.7,
# #                  top_p: float = 0.9,
# #                  **kwargs):
# #         """
# #         Args:
# #             model_id: HuggingFace模型ID或本地路径（如 "Qwen/Qwen-VL-Chat"）
# #             device: 设备 ("cuda", "cpu", "mps"等)
# #             dtype: 数据类型 ("auto", "float16", "bfloat16"等)
# #             trust_remote_code: 是否信任远程代码
# #             load_in_8bit: 是否使用8bit量化
# #             load_in_4bit: 是否使用4bit量化
# #             max_new_tokens: 最大生成token数
# #             temperature: 采样温度
# #             top_p: nucleus采样参数
# #         """
# #         self.model_id = model_id
# #         self.device = device
# #         self.max_new_tokens = max_new_tokens
# #         self.temperature = temperature
# #         self.top_p = top_p
        
# #         try:
# #             from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# #             from PIL import Image
# #             import torch
# #         except ImportError:
# #             raise ImportError(
# #                 "需要安装transformers和torch: pip install transformers torch pillow"
# #             )
        
# #         self.torch = torch
# #         self.Image = Image
        
# #         # 加载tokenizer/processor
# #         try:
# #             self.processor = AutoProcessor.from_pretrained(
# #                 model_id, 
# #                 trust_remote_code=trust_remote_code
# #             )
# #             self.has_processor = True
# #         except:
# #             self.tokenizer = AutoTokenizer.from_pretrained(
# #                 model_id,
# #                 trust_remote_code=trust_remote_code
# #             )
# #             self.has_processor = False
        
# #         # 确定数据类型
# #         if dtype == "auto":
# #             if self.torch.cuda.is_available():
# #                 dtype = self.torch.float16
# #             else:
# #                 dtype = self.torch.float32
# #         elif dtype == "float16":
# #             dtype = self.torch.float16
# #         elif dtype == "bfloat16":
# #             dtype = self.torch.bfloat16
# #         else:
# #             dtype = self.torch.float32
        
# #         # 加载模型
# #         load_kwargs = {
# #             "trust_remote_code": trust_remote_code,
# #             "device_map": device if device != "cpu" else None,
# #         }
        
# #         if device == "cpu":
# #             load_kwargs["torch_dtype"] = dtype
# #         else:
# #             load_kwargs["torch_dtype"] = dtype
        
# #         if load_in_8bit:
# #             load_kwargs["load_in_8bit"] = True
# #         elif load_in_4bit:
# #             load_kwargs["load_in_4bit"] = True
        
# #         self.model = AutoModelForCausalLM.from_pretrained(
# #             model_id,
# #             **load_kwargs
# #         )
# #         self.model.eval()
    
# #     def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
# #         """生成模型响应"""
# #         max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
# #         temperature = kwargs.get("temperature", self.temperature)
# #         top_p = kwargs.get("top_p", self.top_p)
        
# #         # 处理图像
# #         pil_images = None
# #         if images:
# #             pil_images = []
# #             for img_path in images:
# #                 if isinstance(img_path, str):
# #                     if img_path.startswith("http"):
# #                         # 从URL加载
# #                         import requests
# #                         from io import BytesIO
# #                         response = requests.get(img_path)
# #                         pil_images.append(self.Image.open(BytesIO(response.content)))
# #                     elif img_path.startswith("data:image"):
# #                         # Base64编码的图像
# #                         import base64
# #                         from io import BytesIO
# #                         header, encoded = img_path.split(",", 1)
# #                         img_data = base64.b64decode(encoded)
# #                         pil_images.append(self.Image.open(BytesIO(img_data)))
# #                     else:
# #                         # 本地文件路径
# #                         pil_images.append(self.Image.open(img_path))
# #                 else:
# #                     # 假设已经是PIL Image
# #                     pil_images.append(img_path)
        
# #         try:
# #             # 方法1: 使用processor（适用于视觉-语言模型）
# #             if self.has_processor and pil_images:
# #                 inputs = self.processor(
# #                     text=prompt,
# #                     images=pil_images,
# #                     return_tensors="pt"
# #                 )
# #                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
# #                          for k, v in inputs.items()}
                
# #                 with self.torch.no_grad():
# #                     outputs = self.model.generate(
# #                         **inputs,
# #                         max_new_tokens=max_new_tokens,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         do_sample=temperature > 0,
# #                         **kwargs
# #                     )
                
# #                 # 解码输出
# #                 generated_text = self.processor.decode(
# #                     outputs[0],
# #                     skip_special_tokens=True
# #                 )
                
# #                 # 移除输入部分
# #                 if prompt in generated_text:
# #                     generated_text = generated_text.replace(prompt, "").strip()
            
# #             # 方法2: 使用chat接口（如果模型支持）
# #             elif hasattr(self.model, 'chat') and pil_images:
# #                 if self.has_processor:
# #                     response, _ = self.model.chat(
# #                         self.processor,
# #                         query=prompt,
# #                         history=None,
# #                         images=pil_images,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         max_new_tokens=max_new_tokens,
# #                     )
# #                 else:
# #                     response, _ = self.model.chat(
# #                         self.tokenizer,
# #                         query=prompt,
# #                         history=None,
# #                         images=pil_images,
# #                     )
# #                 generated_text = response
            
# #             # 方法3: 纯文本生成
# #             else:
# #                 if self.has_processor:
# #                     inputs = self.processor(text=prompt, return_tensors="pt")
# #                 else:
# #                     inputs = self.tokenizer(prompt, return_tensors="pt")
                
# #                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
# #                          for k, v in inputs.items()}
                
# #                 with self.torch.no_grad():
# #                     outputs = self.model.generate(
# #                         **inputs,
# #                         max_new_tokens=max_new_tokens,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         do_sample=temperature > 0,
# #                         **kwargs
# #                     )
                
# #                 if self.has_processor:
# #                     generated_text = self.processor.decode(
# #                         outputs[0],
# #                         skip_special_tokens=True
# #                     )
# #                 else:
# #                     generated_text = self.tokenizer.decode(
# #                         outputs[0],
# #                         skip_special_tokens=True
# #                     )
                
# #                 # 移除输入部分
# #                 if prompt in generated_text:
# #                     generated_text = generated_text.replace(prompt, "").strip()
            
# #             return {
# #                 "text": generated_text,
# #                 "usage": {"prompt_tokens": 0, "completion_tokens": 0},  # 可以添加实际token统计
# #                 "raw": {"generated_text": generated_text}
# #             }
        
# #         except Exception as e:
# #             raise RuntimeError(f"模型生成失败: {e}")
    
# #     def get_model_info(self) -> Dict[str, Any]:
# #         return {
# #             "name": self.model_id,
# #             "type": "huggingface_local",
# #             "device": self.device
# #         }


# # class HuggingFaceAdapter(BaseModelAdapter):
# #     """HuggingFace本地模型适配器（支持文本和视觉-语言模型）"""
    
# #     def __init__(self, 
# #                  model_id: str,
# #                  device: str = "cuda",
# #                  dtype: str = "auto",
# #                  trust_remote_code: bool = False,
# #                  load_in_8bit: bool = False,
# #                  load_in_4bit: bool = False,
# #                  max_new_tokens: int = 1024,
# #                  temperature: float = 0.7,
# #                  top_p: float = 0.9,
# #                  **kwargs):
# #         """
# #         Args:
# #             model_id: HuggingFace模型ID或本地路径（如 "Qwen/Qwen-VL-Chat"）
# #             device: 设备 ("cuda", "cpu", "mps"等)
# #             dtype: 数据类型 ("auto", "float16", "bfloat16"等)
# #             trust_remote_code: 是否信任远程代码
# #             load_in_8bit: 是否使用8bit量化
# #             load_in_4bit: 是否使用4bit量化
# #             max_new_tokens: 最大生成token数
# #             temperature: 采样温度
# #             top_p: nucleus采样参数
# #         """
# #         self.model_id = model_id
# #         self.device = device
# #         self.max_new_tokens = max_new_tokens
# #         self.temperature = temperature
# #         self.top_p = top_p
        
# #         try:
# #             from transformers import (
# #                 AutoConfig,
# #                 AutoModelForCausalLM,
# #                 AutoModelForVision2Seq,
# #                 BlipForConditionalGeneration,
# #                 AutoTokenizer,
# #                 AutoProcessor
# #             )
# #             from PIL import Image
# #             import torch
# #         except ImportError:
# #             raise ImportError(
# #                 "需要安装transformers和torch: pip install transformers torch pillow"
# #             )
        
# #         self.torch = torch
# #         self.Image = Image
        
# #         # 加载配置以确定模型类型
# #         config = AutoConfig.from_pretrained(
# #             model_id,
# #             trust_remote_code=trust_remote_code
# #         )
        
# #         # 根据模型架构确定模型类型
# #         model_type = config.model_type.lower()
# #         architectures = getattr(config, 'architectures', [])
        
# #         # 判断是否为视觉-语言模型
# #         is_vision_model = any([
# #             'vision' in model_type,
# #             'vl' in model_type,
# #             'blip' in model_type,
# #             'clip' in model_type,
# #             'llava' in model_type,
# #             'qwen-vl' in model_type,
# #             any('Vision' in arch or 'VL' in arch or 'Blip' in arch or 'LLaVA' in arch 
# #                 for arch in architectures)
# #         ])
        
# #         # 加载tokenizer/processor
# #         try:
# #             self.processor = AutoProcessor.from_pretrained(
# #                 model_id, 
# #                 trust_remote_code=trust_remote_code
# #             )
# #             self.has_processor = True
# #         except:
# #             self.tokenizer = AutoTokenizer.from_pretrained(
# #                 model_id,
# #                 trust_remote_code=trust_remote_code
# #             )
# #             self.has_processor = False
        
# #         # 确定数据类型
# #         if dtype == "auto":
# #             if self.torch.cuda.is_available():
# #                 dtype = self.torch.float16
# #             else:
# #                 dtype = self.torch.float32
# #         elif dtype == "float16":
# #             dtype = self.torch.float16
# #         elif dtype == "bfloat16":
# #             dtype = self.torch.bfloat16
# #         else:
# #             dtype = self.torch.float32
        
# #         # 准备加载参数
# #         load_kwargs = {
# #             "trust_remote_code": trust_remote_code,
# #             "device_map": device if device != "cpu" else None,
# #             "torch_dtype": dtype,
# #         }
        
# #         if load_in_8bit:
# #             load_kwargs["load_in_8bit"] = True
# #         elif load_in_4bit:
# #             load_kwargs["load_in_4bit"] = True
        
# #         # 根据模型类型选择合适的加载方式
# #         try:
# #             if 'blip' in model_type:
# #                 # BLIP系列模型
# #                 self.model = BlipForConditionalGeneration.from_pretrained(
# #                     model_id,
# #                     **load_kwargs
# #                 )
# #                 self.model_class = 'blip'
# #             elif is_vision_model:
# #                 # 尝试使用AutoModelForVision2Seq
# #                 try:
# #                     self.model = AutoModelForVision2Seq.from_pretrained(
# #                         model_id,
# #                         **load_kwargs
# #                     )
# #                     self.model_class = 'vision2seq'
# #                 except:
# #                     # 如果失败，尝试AutoModelForCausalLM（某些VL模型使用CausalLM架构）
# #                     self.model = AutoModelForCausalLM.from_pretrained(
# #                         model_id,
# #                         **load_kwargs
# #                     )
# #                     self.model_class = 'causal_lm'
# #             else:
# #                 # 纯文本模型
# #                 self.model = AutoModelForCausalLM.from_pretrained(
# #                     model_id,
# #                     **load_kwargs
# #                 )
# #                 self.model_class = 'causal_lm'
                
# #         except Exception as e:
# #             # 如果所有方法都失败，最后尝试AutoModelForCausalLM
# #             print(f"警告: 使用标准方法加载失败，尝试备用方案: {e}")
# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 model_id,
# #                 **load_kwargs
# #             )
# #             self.model_class = 'causal_lm'
        
# #         self.model.eval()
# #         self.is_vision_model = is_vision_model
    
# #     def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
# #         """生成模型响应"""
# #         max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
# #         temperature = kwargs.get("temperature", self.temperature)
# #         top_p = kwargs.get("top_p", self.top_p)
        
# #         # 处理图像
# #         pil_images = None
# #         if images:
# #             pil_images = []
# #             for img_path in images:
# #                 if isinstance(img_path, str):
# #                     if img_path.startswith("http"):
# #                         # 从URL加载
# #                         import requests
# #                         from io import BytesIO
# #                         response = requests.get(img_path)
# #                         pil_images.append(self.Image.open(BytesIO(response.content)))
# #                     elif img_path.startswith("data:image"):
# #                         # Base64编码的图像
# #                         import base64
# #                         from io import BytesIO
# #                         header, encoded = img_path.split(",", 1)
# #                         img_data = base64.b64decode(encoded)
# #                         pil_images.append(self.Image.open(BytesIO(img_data)))
# #                     else:
# #                         # 本地文件路径
# #                         pil_images.append(self.Image.open(img_path))
# #                 else:
# #                     # 假设已经是PIL Image
# #                     pil_images.append(img_path)
        
# #         try:
# #             # 方法1: 使用processor（适用于视觉-语言模型）
# #             if self.has_processor and pil_images:
# #                 inputs = self.processor(
# #                     text=prompt,
# #                     images=pil_images,
# #                     return_tensors="pt"
# #                 )
# #                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
# #                          for k, v in inputs.items()}
                
# #                 with self.torch.no_grad():
# #                     outputs = self.model.generate(
# #                         **inputs,
# #                         max_new_tokens=max_new_tokens,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         do_sample=temperature > 0,
# #                         **kwargs
# #                     )
                
# #                 # 解码输出
# #                 generated_text = self.processor.decode(
# #                     outputs[0],
# #                     skip_special_tokens=True
# #                 )
                
# #                 # 移除输入部分
# #                 if prompt in generated_text:
# #                     generated_text = generated_text.replace(prompt, "").strip()
            
# #             # 方法2: 使用chat接口（如果模型支持）
# #             elif hasattr(self.model, 'chat') and pil_images:
# #                 if self.has_processor:
# #                     response, _ = self.model.chat(
# #                         self.processor,
# #                         query=prompt,
# #                         history=None,
# #                         images=pil_images,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         max_new_tokens=max_new_tokens,
# #                     )
# #                 else:
# #                     response, _ = self.model.chat(
# #                         self.tokenizer,
# #                         query=prompt,
# #                         history=None,
# #                         images=pil_images,
# #                     )
# #                 generated_text = response
            
# #             # 方法3: 纯文本生成
# #             else:
# #                 if self.has_processor:
# #                     inputs = self.processor(text=prompt, return_tensors="pt")
# #                 else:
# #                     inputs = self.tokenizer(prompt, return_tensors="pt")
                
# #                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
# #                          for k, v in inputs.items()}
                
# #                 with self.torch.no_grad():
# #                     outputs = self.model.generate(
# #                         **inputs,
# #                         max_new_tokens=max_new_tokens,
# #                         temperature=temperature,
# #                         top_p=top_p,
# #                         do_sample=temperature > 0,
# #                         **kwargs
# #                     )
                
# #                 if self.has_processor:
# #                     generated_text = self.processor.decode(
# #                         outputs[0],
# #                         skip_special_tokens=True
# #                     )
# #                 else:
# #                     generated_text = self.tokenizer.decode(
# #                         outputs[0],
# #                         skip_special_tokens=True
# #                     )
                
# #                 # 移除输入部分
# #                 if prompt in generated_text:
# #                     generated_text = generated_text.replace(prompt, "").strip()
            
# #             return {
# #                 "text": generated_text,
# #                 "usage": {"prompt_tokens": 0, "completion_tokens": 0},
# #                 "raw": {"generated_text": generated_text}
# #             }
        
# #         except Exception as e:
# #             raise RuntimeError(f"模型生成失败: {e}")
    
# #     def get_model_info(self) -> Dict[str, Any]:
# #         return {
# #             "name": self.model_id,
# #             "type": "huggingface_local",
# #             "device": self.device,
# #             "model_class": self.model_class,
# #             "is_vision_model": self.is_vision_model
# #         }


# class HuggingFaceAdapter(BaseModelAdapter):
#     """HuggingFace本地模型适配器（支持文本和视觉-语言模型）"""
    
#     def __init__(self, 
#                  model_id: str,
#                  device: str = "cuda",
#                  dtype: str = "auto",
#                  trust_remote_code: bool = False,
#                  load_in_8bit: bool = False,
#                  load_in_4bit: bool = False,
#                  max_new_tokens: int = 1024,
#                  temperature: float = 0.7,
#                  top_p: float = 0.9,
#                  **kwargs):
#         """
#         Args:
#             model_id: HuggingFace模型ID或本地路径（如 "Qwen/Qwen-VL-Chat"）
#             device: 设备 ("cuda", "cpu", "mps"等)
#             dtype: 数据类型 ("auto", "float16", "bfloat16"等)
#             trust_remote_code: 是否信任远程代码
#             load_in_8bit: 是否使用8bit量化
#             load_in_4bit: 是否使用4bit量化
#             max_new_tokens: 最大生成token数
#             temperature: 采样温度
#             top_p: nucleus采样参数
#         """
#         self.model_id = model_id
#         self.device = device
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.top_p = top_p
        
#         try:
#             from transformers import (
#                 AutoConfig,
#                 AutoModelForCausalLM,
#                 AutoModelForVision2Seq,
#                 BlipForConditionalGeneration,
#                 AutoTokenizer,
#                 AutoProcessor
#             )
#             from PIL import Image
#             import torch
#         except ImportError:
#             raise ImportError(
#                 "需要安装transformers和torch: pip install transformers torch pillow"
#             )
        
#         self.torch = torch
#         self.Image = Image
        
#         # 加载配置以确定模型类型
#         config = AutoConfig.from_pretrained(
#             model_id,
#             trust_remote_code=trust_remote_code
#         )
        
#         # 根据模型架构确定模型类型
#         model_type = config.model_type.lower()
#         architectures = getattr(config, 'architectures', [])
        
#         # 判断是否为视觉-语言模型
#         is_vision_model = any([
#             'vision' in model_type,
#             'vl' in model_type,
#             'blip' in model_type,
#             'clip' in model_type,
#             'llava' in model_type,
#             'qwen-vl' in model_type,
#             any('Vision' in arch or 'VL' in arch or 'Blip' in arch or 'LLaVA' in arch 
#                 for arch in architectures)
#         ])
        
#         # 加载tokenizer/processor
#         try:
#             self.processor = AutoProcessor.from_pretrained(
#                 model_id, 
#                 trust_remote_code=trust_remote_code
#             )
#             self.has_processor = True
#         except:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_id,
#                 trust_remote_code=trust_remote_code
#             )
#             self.has_processor = False
        
#         # 确定数据类型
#         if dtype == "auto":
#             if self.torch.cuda.is_available():
#                 dtype_obj = self.torch.float16
#             else:
#                 dtype_obj = self.torch.float32
#         elif dtype == "float16":
#             dtype_obj = self.torch.float16
#         elif dtype == "bfloat16":
#             dtype_obj = self.torch.bfloat16
#         else:
#             dtype_obj = self.torch.float32
        
#         # 检查是否安装了accelerate（用于device_map）
#         try:
#             import accelerate
#             has_accelerate = True
#         except ImportError:
#             has_accelerate = False
        
#         # 准备加载参数
#         load_kwargs = {
#             "trust_remote_code": trust_remote_code,
#             "torch_dtype": dtype_obj,
#         }
        
#         # 只在有accelerate或device为cpu时设置device_map
#         if device == "cpu":
#             # CPU模式不需要device_map
#             pass
#         elif has_accelerate:
#             load_kwargs["device_map"] = device
#         else:
#             # 没有accelerate，手动指定设备
#             print(f"警告: 未安装accelerate，将在加载后手动移动模型到 {device}")
        
#         if load_in_8bit:
#             load_kwargs["load_in_8bit"] = True
#         elif load_in_4bit:
#             load_kwargs["load_in_4bit"] = True
        
#         # 根据模型类型选择合适的加载方式
#         self.model = None
#         self.model_class = None
        
#         try:
#             if 'blip' in model_type:
#                 # BLIP系列模型
#                 self.model = BlipForConditionalGeneration.from_pretrained(
#                     model_id,
#                     **load_kwargs
#                 )
#                 self.model_class = 'blip'
#             elif is_vision_model:
#                 # 尝试使用AutoModelForVision2Seq
#                 try:
#                     self.model = AutoModelForVision2Seq.from_pretrained(
#                         model_id,
#                         **load_kwargs
#                     )
#                     self.model_class = 'vision2seq'
#                 except Exception as e:
#                     # 某些VL模型使用CausalLM架构，但需要确保不是BLIP
#                     if 'blip' not in model_type:
#                         self.model = AutoModelForCausalLM.from_pretrained(
#                             model_id,
#                             **load_kwargs
#                         )
#                         self.model_class = 'causal_lm'
#                     else:
#                         raise e
#             else:
#                 # 纯文本模型
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_id,
#                     **load_kwargs
#                 )
#                 self.model_class = 'causal_lm'
                
#         except Exception as e:
#             raise RuntimeError(f"模型加载失败: {e}\n提示: 如果是device_map相关错误，请安装accelerate: pip install accelerate")
        
#         # 如果没有使用device_map，手动移动模型
#         if device != "cpu" and not has_accelerate and "device_map" not in load_kwargs:
#             self.model = self.model.to(device)
        
#         self.model.eval()
#         self.is_vision_model = is_vision_model
        
#         # 用于跟踪是否是首次调用
#         self._first_call = True
#         self._first_output_printed = False
    
#     def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
#         """生成模型响应"""
#         max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
#         temperature = kwargs.get("temperature", self.temperature)
#         top_p = kwargs.get("top_p", self.top_p)
        
#         # 处理图像
#         pil_images = None
#         if images:
#             pil_images = []
#             for img_path in images:
#                 if isinstance(img_path, str):
#                     if img_path.startswith("http"):
#                         # 从URL加载
#                         import requests
#                         from io import BytesIO
#                         response = requests.get(img_path)
#                         pil_images.append(self.Image.open(BytesIO(response.content)))
#                     elif img_path.startswith("data:image"):
#                         # Base64编码的图像
#                         import base64
#                         from io import BytesIO
#                         header, encoded = img_path.split(",", 1)
#                         img_data = base64.b64decode(encoded)
#                         pil_images.append(self.Image.open(BytesIO(img_data)))
#                     else:
#                         # 本地文件路径
#                         pil_images.append(self.Image.open(img_path))
#                 else:
#                     # 假设已经是PIL Image
#                     pil_images.append(img_path)
        
#         # 用于保存输出信息（首次调用时使用）
#         outputs_info = None
#         inputs_info = None
#         generation_method = None
        
#         try:
#             # 方法1: 使用processor（适用于视觉-语言模型）
#             if self.has_processor and pil_images:
#                 generation_method = "processor_with_images"
                
#                 # 对于BLIP VQA模型，需要特殊处理
#                 # BLIP VQA应该从图像特征生成，而不是从编码的问题
#                 # 我们需要只传入图像和问题文本，让模型自己处理
#                 if self.model_class == 'blip':
#                     # BLIP VQA的特殊处理：只传入图像和问题文本，不传入预编码的input_ids
#                     inputs = self.processor(
#                         images=pil_images,
#                         text=prompt,
#                         return_tensors="pt",
#                         padding=True
#                     )
#                     inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                              for k, v in inputs.items()}
                    
#                     # 对于BLIP VQA，我们不使用预编码的input_ids作为生成起点
#                     # 而是让模型从图像特征开始生成
#                     # 所以我们需要创建一个decoder_input_ids，通常以[BOS] token开始
#                     if 'input_ids' in inputs:
#                         # 获取decoder的起始token（通常是[BOS] token）
#                         decoder_start_token_id = self.model.config.decoder_start_token_id if hasattr(self.model.config, 'decoder_start_token_id') else None
#                         if decoder_start_token_id is None:
#                             # 如果没有设置，尝试使用processor的tokenizer的bos_token_id
#                             if hasattr(self.processor, 'tokenizer'):
#                                 decoder_start_token_id = self.processor.tokenizer.bos_token_id
#                             elif hasattr(self.processor, 'decoder_tokenizer'):
#                                 decoder_start_token_id = self.processor.decoder_tokenizer.bos_token_id
                        
#                         # 创建decoder_input_ids，从BOS token开始
#                         batch_size = inputs['pixel_values'].shape[0] if hasattr(inputs['pixel_values'], 'shape') else 1
                        
#                         # 获取decoder的起始token ID
#                         if decoder_start_token_id is None:
#                             # 尝试其他方式获取BOS token
#                             if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'bos_token_id'):
#                                 decoder_start_token_id = self.processor.tokenizer.bos_token_id
#                             elif hasattr(self.processor, 'decoder_tokenizer') and hasattr(self.processor.decoder_tokenizer, 'bos_token_id'):
#                                 decoder_start_token_id = self.processor.decoder_tokenizer.bos_token_id
#                             elif hasattr(self.model.config, 'bos_token_id'):
#                                 decoder_start_token_id = self.model.config.bos_token_id
                        
#                         # 如果还是没有找到，尝试pad_token_id
#                         if decoder_start_token_id is None:
#                             pad_token_id = getattr(self.model.config, 'pad_token_id', None)
#                             if pad_token_id is None:
#                                 # 最后使用0作为默认值
#                                 decoder_start_token_id = 0
#                             else:
#                                 decoder_start_token_id = pad_token_id
                        
#                         # 确保decoder_start_token_id是整数
#                         decoder_start_token_id = int(decoder_start_token_id)
                        
#                         # 创建decoder_input_ids
#                         decoder_input_ids = self.torch.full(
#                             (batch_size, 1), 
#                             decoder_start_token_id, 
#                             dtype=self.torch.long, 
#                             device=self.device
#                         )
                        
#                         # 移除input_ids，只保留pixel_values和其他必要的输入
#                         # 使用decoder_input_ids代替input_ids
#                         inputs = {k: v for k, v in inputs.items() if k != 'input_ids'}
#                         inputs['decoder_input_ids'] = decoder_input_ids
#                         input_length = 0  # decoder从空开始生成
#                     else:
#                         input_length = 0
#                 else:
#                     # 非BLIP模型，使用标准处理
#                     inputs = self.processor(
#                         text=prompt,
#                         images=pil_images,
#                         return_tensors="pt"
#                     )
#                     inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                              for k, v in inputs.items()}
#                     input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                
#                 # 首次调用时保存并打印输入信息
#                 if self._first_call:
#                     inputs_info = self._extract_inputs_info(inputs, pil_images)
#                     self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
#                     print("\n📥 处理后的输入结构:")
#                     self._print_inputs_structure(inputs_info)
#                     print(f"  • 输入长度（用于解码）: {input_length}")
                
#                 with self.torch.no_grad():
#                     # 生成参数
#                     generate_kwargs = {
#                         "max_new_tokens": max_new_tokens,
#                         "temperature": temperature,
#                         "top_p": top_p,
#                         "do_sample": temperature > 0,
#                     }
#                     # 添加其他kwargs，但避免覆盖重要参数
#                     for key, value in kwargs.items():
#                         if key not in generate_kwargs:
#                             generate_kwargs[key] = value
                    
#                     if self._first_call:
#                         print(f"\n🔍 调试信息 - Generate调用参数:")
#                         print(f"  • inputs键: {list(inputs.keys())}")
#                         print(f"  • generate_kwargs: {generate_kwargs}")
#                         print(f"  • input_length: {input_length} (decoder起始长度)")
                    
#                     outputs = self.model.generate(
#                         **inputs,
#                         **generate_kwargs
#                     )
                    
#                     if self._first_call:
#                         print(f"  • generate返回的outputs类型: {type(outputs)}")
#                         print(f"  • generate返回的outputs形状: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
#                         if hasattr(outputs, '__len__') and len(outputs) > 0:
#                             print(f"  • generate返回的outputs[0]长度: {len(outputs[0])}")
#                         print(f"  • 预期长度: max_new_tokens({max_new_tokens})")
#                         print(f"  • 实际生成长度: {len(outputs[0]) if hasattr(outputs, '__len__') and len(outputs) > 0 else 'N/A'}")
                
#                 # 首次调用时保存输出信息
#                 if self._first_call:
#                     outputs_info = self._extract_outputs_info(outputs, input_length)
#                     # 添加详细的输出调试信息
#                     print(f"\n🔍 调试信息 - Generate输出检查:")
#                     print(f"  • outputs类型: {type(outputs)}")
#                     print(f"  • outputs形状: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
#                     print(f"  • outputs[0]形状: {outputs[0].shape if hasattr(outputs[0], 'shape') else 'N/A'}")
#                     print(f"  • outputs[0]长度: {len(outputs[0]) if hasattr(outputs[0], '__len__') else 'N/A'}")
#                     print(f"  • input_length: {input_length}")
#                     print(f"  • 是否有新生成内容: {len(outputs[0]) > input_length}")
#                     if len(outputs[0]) > input_length:
#                         print(f"  • 新生成的token数量: {len(outputs[0]) - input_length}")
#                         print(f"  • 新生成的token IDs: {outputs[0][input_length:].tolist()}")
#                     else:
#                         print(f"  ⚠️  警告: 输出长度({len(outputs[0])}) <= 输入长度({input_length})，模型可能没有生成新内容！")
#                         print(f"  • 完整的outputs[0]: {outputs[0].tolist()}")
#                         print(f"  • 输入的input_ids: {inputs['input_ids'][0].tolist()}")
                
#                 # 对于BLIP模型，input_length为0，直接解码整个输出
#                 # 对于其他模型，如果input_length > 0，只解码新生成的部分
#                 if input_length > 0 and len(outputs[0]) > input_length:
#                     # 只解码新生成的token IDs
#                     generated_ids = outputs[0][input_length:]
#                     generated_text = self.processor.decode(
#                         generated_ids,
#                         skip_special_tokens=True
#                     )
#                     if self._first_call:
#                         print(f"  • 解码的新生成文本: '{generated_text}'")
#                 else:
#                     # 对于BLIP模型或无法确定输入长度的情况，解码整个输出
#                     generated_text = self.processor.decode(
#                         outputs[0],
#                         skip_special_tokens=True
#                     )
#                     if self._first_call:
#                         print(f"  • 完整解码文本: '{generated_text}'")
#                         if input_length == 0:
#                             print(f"  • 注意: input_length=0，使用完整解码（适用于BLIP等从decoder开始生成的模型）")
                    
#                     # 如果input_length > 0，说明可能需要移除输入部分
#                     # 但对于BLIP，input_length=0，所以不需要移除
#                     if input_length > 0 and prompt in generated_text:
#                         # 找到prompt在文本中的位置并移除
#                         prompt_pos = generated_text.find(prompt)
#                         if prompt_pos == 0:
#                             # prompt在开头，直接移除
#                             generated_text = generated_text[len(prompt):].strip()
#                             if self._first_call:
#                                 print(f"  • 移除开头的prompt后: '{generated_text}'")
#                         else:
#                             # prompt在中间或末尾，尝试移除
#                             generated_text = generated_text.replace(prompt, "").strip()
#                             if self._first_call:
#                                 print(f"  • 替换prompt后: '{generated_text}'")
                    
#                     # 如果生成的文本为空，说明可能有问题
#                     if not generated_text and self._first_call:
#                         print(f"  ⚠️  警告: 生成的文本为空！可能需要检查模型generate的参数或模型本身")
            
#             # 方法2: 使用chat接口（如果模型支持）
#             elif hasattr(self.model, 'chat') and pil_images:
#                 generation_method = "chat_interface"
#                 # 首次调用时保存并打印输入信息
#                 if self._first_call:
#                     inputs_info = {
#                         "method": "chat",
#                         "prompt": prompt,
#                         "images_count": len(pil_images) if pil_images else 0,
#                         "image_sizes": [img.size for img in pil_images] if pil_images else [],
#                         "has_processor": self.has_processor
#                     }
#                     self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
#                     print("\n📥 Chat接口输入信息:")
#                     print(f"  • 图像数量: {inputs_info['images_count']}")
#                     print(f"  • 图像尺寸: {inputs_info['image_sizes']}")
#                     print(f"  • 使用Processor: {inputs_info['has_processor']}")
                
#                 if self.has_processor:
#                     response, _ = self.model.chat(
#                         self.processor,
#                         query=prompt,
#                         history=None,
#                         images=pil_images,
#                         temperature=temperature,
#                         top_p=top_p,
#                         max_new_tokens=max_new_tokens,
#                     )
#                 else:
#                     response, _ = self.model.chat(
#                         self.tokenizer,
#                         query=prompt,
#                         history=None,
#                         images=pil_images,
#                     )
#                 generated_text = response
                
#                 # 首次调用时保存输出信息
#                 if self._first_call:
#                     outputs_info = {
#                         "method": "chat",
#                         "response_type": type(response).__name__,
#                         "response_length": len(response) if isinstance(response, str) else "N/A"
#                     }
            
#             # 方法3: 纯文本生成
#             else:
#                 generation_method = "text_only"
#                 if self.has_processor:
#                     inputs = self.processor(text=prompt, return_tensors="pt")
#                 else:
#                     inputs = self.tokenizer(prompt, return_tensors="pt")
                
#                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                          for k, v in inputs.items()}
                
#                 # 首次调用时保存并打印输入信息
#                 if self._first_call:
#                     inputs_info = self._extract_inputs_info(inputs, None)
#                     self._print_first_call_info(prompt, None, max_new_tokens, temperature, top_p, kwargs)
#                     print("\n📥 处理后的输入结构:")
#                     self._print_inputs_structure(inputs_info)
                
#                 # 保存输入长度，用于后续只解码新生成的部分
#                 input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                
#                 with self.torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=temperature > 0,
#                         **kwargs
#                     )
                
#                 # 首次调用时保存输出信息
#                 if self._first_call:
#                     outputs_info = self._extract_outputs_info(outputs, input_length)
                
#                 # 只解码新生成的部分（排除输入部分）
#                 if input_length > 0 and len(outputs[0]) > input_length:
#                     # 只解码新生成的token IDs
#                     generated_ids = outputs[0][input_length:]
#                     if self.has_processor:
#                         generated_text = self.processor.decode(
#                             generated_ids,
#                             skip_special_tokens=True
#                         )
#                     else:
#                         generated_text = self.tokenizer.decode(
#                             generated_ids,
#                             skip_special_tokens=True
#                         )
#                 else:
#                     # 如果无法确定输入长度，解码整个输出然后移除输入部分
#                     if self.has_processor:
#                         generated_text = self.processor.decode(
#                             outputs[0],
#                             skip_special_tokens=True
#                         )
#                     else:
#                         generated_text = self.tokenizer.decode(
#                             outputs[0],
#                             skip_special_tokens=True
#                         )
#                     # 移除输入部分（更精确的方法）
#                     if prompt in generated_text:
#                         # 找到prompt在文本中的位置并移除
#                         prompt_pos = generated_text.find(prompt)
#                         if prompt_pos == 0:
#                             # prompt在开头，直接移除
#                             generated_text = generated_text[len(prompt):].strip()
#                         else:
#                             # prompt在中间或末尾，尝试移除
#                             generated_text = generated_text.replace(prompt, "").strip()
            
#             result = {
#                 "text": generated_text,
#                 "usage": {"prompt_tokens": 0, "completion_tokens": 0},
#                 "raw": {"generated_text": generated_text}
#             }
            
#             # 首次调用时输出详细信息
#             if self._first_call:
#                 self._print_first_output_info(result, outputs_info, generation_method)
#                 self._first_call = False
#                 self._first_output_printed = True
            
#             return result
        
#         except Exception as e:
#             raise RuntimeError(f"模型生成失败: {e}")
    
#     def _print_inputs_structure(self, inputs_info: Dict):
#         """打印输入结构信息"""
#         if "keys" in inputs_info:
#             print(f"  • 输入键: {inputs_info['keys']}")
#         if "tensor_info" in inputs_info:
#             print(f"  • Tensor信息:")
#             for key, info in inputs_info["tensor_info"].items():
#                 if "shape" in info:
#                     print(f"    - {key}: shape={info['shape']}, dtype={info['dtype']}, device={info['device']}")
#                 else:
#                     print(f"    - {key}: {info}")
#         if "images" in inputs_info:
#             print(f"  • 图像信息:")
#             for i, img_info in enumerate(inputs_info["images"]):
#                 print(f"    - 图像 {i+1}: {img_info}")
    
#     def _print_first_call_info(self, prompt: str, images: Optional[List], max_new_tokens: int, 
#                                temperature: float, top_p: float, kwargs: Dict):
#         """打印首次调用时的详细信息"""
#         print("\n" + "="*80)
#         print("🔍 首次模型调用 - 输入信息")
#         print("="*80)
        
#         # 模型信息
#         print("\n📦 模型信息:")
#         print(f"  • 模型ID: {self.model_id}")
#         print(f"  • 模型类型: {self.model_class}")
#         print(f"  • 设备: {self.device}")
#         print(f"  • 是否视觉模型: {self.is_vision_model}")
#         print(f"  • 使用Processor: {self.has_processor}")
#         if hasattr(self, 'model'):
#             print(f"  • 模型类: {type(self.model).__name__}")
#             if hasattr(self.model, 'config'):
#                 config = self.model.config
#                 print(f"  • 模型配置类型: {type(config).__name__}")
#                 if hasattr(config, 'vocab_size'):
#                     print(f"  • 词汇表大小: {config.vocab_size}")
#                 if hasattr(config, 'max_position_embeddings'):
#                     print(f"  • 最大位置编码: {config.max_position_embeddings}")
        
#         # 输入信息
#         print("\n📥 输入信息:")
#         print(f"  • Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
#         print(f"  • Prompt长度: {len(prompt)} 字符")
#         if images:
#             print(f"  • 图像数量: {len(images)}")
#             for i, img in enumerate(images):
#                 if isinstance(img, str):
#                     print(f"    - 图像 {i+1}: {img[:100]}{'...' if len(img) > 100 else ''}")
#                 elif hasattr(img, 'size'):
#                     print(f"    - 图像 {i+1}: PIL Image, 尺寸: {img.size}, 模式: {img.mode}")
#                 else:
#                     print(f"    - 图像 {i+1}: {type(img).__name__}")
#         else:
#             print(f"  • 图像数量: 0")
        
#         # 生成参数
#         print("\n⚙️  生成参数:")
#         print(f"  • max_new_tokens: {max_new_tokens}")
#         print(f"  • temperature: {temperature}")
#         print(f"  • top_p: {top_p}")
#         if kwargs:
#             print(f"  • 其他参数: {kwargs}")
        
#         print("="*80 + "\n")
    
#     def _extract_inputs_info(self, inputs: Dict, images: Optional[List]) -> Dict:
#         """提取输入信息的结构化数据"""
#         info = {
#             "keys": list(inputs.keys()),
#             "tensor_info": {}
#         }
        
#         for key, value in inputs.items():
#             if isinstance(value, self.torch.Tensor):
#                 info["tensor_info"][key] = {
#                     "shape": list(value.shape),
#                     "dtype": str(value.dtype),
#                     "device": str(value.device),
#                     "requires_grad": value.requires_grad
#                 }
#             else:
#                 info["tensor_info"][key] = {
#                     "type": type(value).__name__,
#                     "value": str(value)[:100] if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
#                 }
        
#         if images:
#             info["images"] = []
#             for img in images:
#                 if hasattr(img, 'size'):
#                     info["images"].append({
#                         "type": "PIL.Image",
#                         "size": img.size,
#                         "mode": img.mode
#                     })
#                 else:
#                     info["images"].append({"type": type(img).__name__})
        
#         return info
    
#     def _extract_outputs_info(self, outputs, input_length: int) -> Dict:
#         """提取输出信息的结构化数据"""
#         info = {
#             "output_type": type(outputs).__name__,
#             "input_length": input_length
#         }
        
#         if isinstance(outputs, self.torch.Tensor):
#             info["shape"] = list(outputs.shape)
#             info["dtype"] = str(outputs.dtype)
#             info["device"] = str(outputs.device)
#         elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
#             first_output = outputs[0] if isinstance(outputs, list) else outputs[0]
#             if isinstance(first_output, self.torch.Tensor):
#                 info["first_output_shape"] = list(first_output.shape)
#                 info["first_output_dtype"] = str(first_output.dtype)
#                 info["total_length"] = first_output.shape[0] if len(first_output.shape) > 0 else "N/A"
#                 info["generated_length"] = first_output.shape[0] - input_length if input_length > 0 else "N/A"
        
#         return info
    
#     def _print_first_output_info(self, result: Dict, outputs_info: Optional[Dict], generation_method: Optional[str]):
#         """打印首次调用时的输出信息"""
#         print("\n" + "="*80)
#         print("📤 首次模型调用 - 输出信息")
#         print("="*80)
        
#         # 生成方法
#         if generation_method:
#             print(f"\n🔧 使用的生成方法: {generation_method}")
        
#         # 输出结构
#         print("\n📊 输出结构:")
#         print(f"  • 返回类型: {type(result).__name__}")
#         print(f"  • 返回键: {list(result.keys())}")
        
#         # 生成的文本
#         if "text" in result:
#             text = result["text"]
#             print(f"\n💬 生成的文本:")
#             print(f"  • 内容: {text[:200]}{'...' if len(text) > 200 else ''}")
#             print(f"  • 长度: {len(text)} 字符")
        
#         # 原始输出信息
#         if outputs_info:
#             print(f"\n🔍 原始输出信息:")
#             if "output_type" in outputs_info:
#                 print(f"  • 输出类型: {outputs_info['output_type']}")
#             if "shape" in outputs_info:
#                 print(f"  • 形状: {outputs_info['shape']}")
#             if "first_output_shape" in outputs_info:
#                 print(f"  • 第一个输出形状: {outputs_info['first_output_shape']}")
#             if "input_length" in outputs_info:
#                 print(f"  • 输入长度: {outputs_info['input_length']}")
#             if "generated_length" in outputs_info:
#                 print(f"  • 生成长度: {outputs_info['generated_length']}")
        
#         # Usage信息
#         if "usage" in result:
#             print(f"\n📈 Token使用:")
#             for key, value in result["usage"].items():
#                 print(f"  • {key}: {value}")
        
#         # Raw信息（简要）
#         if "raw" in result:
#             raw = result["raw"]
#             print(f"\n📦 Raw响应:")
#             print(f"  • 类型: {type(raw).__name__}")
#             if isinstance(raw, dict):
#                 print(f"  • 键: {list(raw.keys())}")
        
#         print("="*80 + "\n")
    
#     def get_model_info(self) -> Dict[str, Any]:
#         return {
#             "name": self.model_id,
#             "type": "huggingface_local",
#             "device": self.device,
#             "model_class": self.model_class,
#             "is_vision_model": self.is_vision_model
#         }


# # HuggingFace Hub Inference API适配器
# class HuggingFaceHubAdapter(BaseModelAdapter):
#     """HuggingFace Hub Inference API适配器"""
    
#     def __init__(self, 
#                  model_id: str,
#                  api_token: Optional[str] = None,
#                  api_url: Optional[str] = None,
#                  timeout: float = 30.0,
#                  **kwargs):
#         """
#         Args:
#             model_id: HuggingFace模型ID
#             api_token: HuggingFace API token（可选，用于私有模型）
#             api_url: 自定义API URL（可选，默认使用HuggingFace Inference API）
#             timeout: 请求超时时间
#         """
#         self.model_id = model_id
#         self.api_token = api_token
#         self.timeout = timeout
        
#         if api_url:
#             self.api_url = api_url.rstrip("/")
#         else:
#             self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
#         try:
#             import requests
#             self.requests = requests
#         except ImportError:
#             raise ImportError("需要安装requests: pip install requests")
    
#     def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
#         """生成模型响应"""
#         headers = {}
#         if self.api_token:
#             headers["Authorization"] = f"Bearer {self.api_token}"
        
#         # 构建请求payload
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 1024)),
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_p": kwargs.get("top_p", 0.9),
#             }
#         }
        
#         # 处理图像
#         if images:
#             # HuggingFace Inference API支持图像输入
#             # 对于视觉-语言模型，inputs可以是字典
#             if len(images) == 1:
#                 img_path = images[0]
#                 if img_path.startswith("http") or img_path.startswith("data:image"):
#                     payload["inputs"] = {
#                         "text": prompt,
#                         "image": img_path
#                     }
#                 else:
#                     # 读取本地图像并转换为base64
#                     from PIL import Image
#                     import base64
#                     from io import BytesIO
                    
#                     img = Image.open(img_path)
#                     buffered = BytesIO()
#                     img.save(buffered, format="JPEG")
#                     img_base64 = base64.b64encode(buffered.getvalue()).decode()
#                     payload["inputs"] = {
#                         "text": prompt,
#                         "image": f"data:image/jpeg;base64,{img_base64}"
#                     }
        
#         try:
#             response = self.requests.post(
#                 self.api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
#             result = response.json()
            
#             # 解析响应（HuggingFace API返回格式可能不同）
#             if isinstance(result, list) and len(result) > 0:
#                 if "generated_text" in result[0]:
#                     text = result[0]["generated_text"]
#                 elif "answer" in result[0]:
#                     text = result[0]["answer"]
#                 else:
#                     text = str(result[0])
#             elif isinstance(result, dict):
#                 text = result.get("generated_text", result.get("answer", str(result)))
#             else:
#                 text = str(result)
            
#             # 移除输入部分
#             if prompt in text:
#                 text = text.replace(prompt, "").strip()
            
#             return {
#                 "text": text,
#                 "usage": {},
#                 "raw": result
#             }
        
#         except Exception as e:
#             raise RuntimeError(f"HuggingFace API调用失败: {e}")
    
#     def get_model_info(self) -> Dict[str, Any]:
#         return {
#             "name": self.model_id,
#             "type": "huggingface_hub_api",
#             "api_url": self.api_url
#         }


# # 注册默认适配器
# ModelAdapterFactory.register("openai", OpenAIAdapter)
# ModelAdapterFactory.register("huggingface", HuggingFaceAdapter)
# ModelAdapterFactory.register("hf", HuggingFaceAdapter)  # 简写
# ModelAdapterFactory.register("huggingface_hub", HuggingFaceHubAdapter)
# ModelAdapterFactory.register("hf_hub", HuggingFaceHubAdapter)  # 简写








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型适配器：统一接口，允许用户接入自己的模型
"""

import abc
from typing import Any, Dict, List, Optional, Union


class BaseModelAdapter(abc.ABC):
    """模型适配器基类"""
    
    @abc.abstractmethod
    def generate(self, 
                prompt: str,
                images: Optional[List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        生成模型响应
        
        Args:
            prompt: 文本提示
            images: 图像路径列表或base64编码列表
            **kwargs: 其他参数（temperature, max_tokens等）
        
        Returns:
            {
                "text": "模型输出文本",
                "usage": {...},  # 可选的token使用信息
                "raw": {...}     # 可选的原始响应
            }
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            {
                "name": "模型名称",
                "type": "模型类型",
                "version": "版本号"（可选）
            }
        """
        raise NotImplementedError


class ModelAdapterFactory:
    """模型适配器工厂"""
    
    _adapters = {}
    
    @classmethod
    def register(cls, name: str, adapter_class):
        """注册适配器类"""
        cls._adapters[name] = adapter_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModelAdapter:
        """创建适配器实例"""
        if name not in cls._adapters:
            raise ValueError(f"未知的适配器类型: {name}")
        return cls._adapters[name](**kwargs)
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """列出所有已注册的适配器"""
        return list(cls._adapters.keys())


# 示例：OpenAI兼容的适配器
class OpenAIAdapter(BaseModelAdapter):
    """OpenAI兼容的API适配器"""
    
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.kwargs = kwargs
    
    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        import requests
        
        messages = [{"role": "user", "content": []}]
        
        # 添加文本
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        # 添加图像（如果提供）
        if images:
            for img in images:
                if img.startswith("http"):
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": img}})
                else:
                    # 假设是base64
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=kwargs.get("timeout", 30)
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "text": result["choices"][0]["message"]["content"],
            "usage": result.get("usage", {}),
            "raw": result
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "type": "openai_compatible",
            "base_url": self.base_url
        }


class HuggingFaceAdapter(BaseModelAdapter):
    """HuggingFace本地模型适配器（支持文本和视觉-语言模型）"""
    
    def __init__(self, 
                 model_id: str,
                 device: str = "cuda",
                 dtype: str = "auto",
                 trust_remote_code: bool = False,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **kwargs):
        """
        Args:
            model_id: HuggingFace模型ID或本地路径（如 "Qwen/Qwen-VL-Chat"）
            device: 设备 ("cuda", "cpu", "mps"等)
            dtype: 数据类型 ("auto", "float16", "bfloat16"等)
            trust_remote_code: 是否信任远程代码
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
        """
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForVision2Seq,
                BlipForConditionalGeneration,
                AutoTokenizer,
                AutoProcessor
            )
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError(
                "需要安装transformers和torch: pip install transformers torch pillow"
            )
        
        self.torch = torch
        self.Image = Image
        
        # 加载配置以确定模型类型
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        # 根据模型架构确定模型类型
        model_type = config.model_type.lower()
        architectures = getattr(config, 'architectures', [])
        
        # 判断是否为视觉-语言模型
        is_vision_model = any([
            'vision' in model_type,
            'vl' in model_type,
            'blip' in model_type,
            'clip' in model_type,
            'llava' in model_type,
            'qwen-vl' in model_type,
            any('Vision' in arch or 'VL' in arch or 'Blip' in arch or 'LLaVA' in arch 
                for arch in architectures)
        ])
        
        # 加载tokenizer/processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=trust_remote_code
            )
            self.has_processor = True
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code
            )
            self.has_processor = False
        
        # 确定数据类型
        if dtype == "auto":
            if self.torch.cuda.is_available():
                dtype_obj = self.torch.float16
            else:
                dtype_obj = self.torch.float32
        elif dtype == "float16":
            dtype_obj = self.torch.float16
        elif dtype == "bfloat16":
            dtype_obj = self.torch.bfloat16
        else:
            dtype_obj = self.torch.float32
        
        # 检查是否安装了accelerate（用于device_map）
        try:
            import accelerate
            has_accelerate = True
        except ImportError:
            has_accelerate = False
        
        # 准备加载参数
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype_obj,
        }
        
        # 只在有accelerate或device为cpu时设置device_map
        if device == "cpu":
            # CPU模式不需要device_map
            pass
        elif has_accelerate:
            load_kwargs["device_map"] = device
        else:
            # 没有accelerate，手动指定设备
            print(f"警告: 未安装accelerate，将在加载后手动移动模型到 {device}")
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        # 根据模型类型选择合适的加载方式
        self.model = None
        self.model_class = None
        
        try:
            if 'blip' in model_type:
                # BLIP系列模型 - 根据任务选择正确的模型类
                # 关键：BLIP有多个变体，需要根据模型名称判断
                
                # 检查是否是 VQA 模型
                if 'vqa' in model_id.lower():
                    # 使用 BlipForQuestionAnswering（VQA专用）
                    from transformers import BlipForQuestionAnswering
                    self.model = BlipForQuestionAnswering.from_pretrained(
                        model_id,
                        **load_kwargs
                    )
                    self.model_class = 'blip_vqa'
                    print(f"✅ 加载 BLIP VQA 模型: BlipForQuestionAnswering")
                else:
                    # 使用 BlipForConditionalGeneration（Caption/生成任务）
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        model_id,
                        **load_kwargs
                    )
                    self.model_class = 'blip_caption'
                    print(f"✅ 加载 BLIP Caption 模型: BlipForConditionalGeneration")
                    
            elif is_vision_model:
                # 尝试使用AutoModelForVision2Seq
                try:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        **load_kwargs
                    )
                    self.model_class = 'vision2seq'
                except Exception as e:
                    # 某些VL模型使用CausalLM架构，但需要确保不是BLIP
                    if 'blip' not in model_type:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            **load_kwargs
                        )
                        self.model_class = 'causal_lm'
                    else:
                        raise e
            else:
                # 纯文本模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **load_kwargs
                )
                self.model_class = 'causal_lm'
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}\n提示: 如果是device_map相关错误，请安装accelerate: pip install accelerate")
        
        # 如果没有使用device_map，手动移动模型
        if device != "cpu" and not has_accelerate and "device_map" not in load_kwargs:
            self.model = self.model.to(device)
        
        self.model.eval()
        self.is_vision_model = is_vision_model
        
        # 用于跟踪是否是首次调用
        self._first_call = True
        self._first_output_printed = False
    
    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """生成模型响应"""
        max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # 处理图像
        pil_images = None
        if images:
            pil_images = []
            for img_path in images:
                if isinstance(img_path, str):
                    if img_path.startswith("http"):
                        # 从URL加载
                        import requests
                        from io import BytesIO
                        response = requests.get(img_path)
                        pil_images.append(self.Image.open(BytesIO(response.content)))
                    elif img_path.startswith("data:image"):
                        # Base64编码的图像
                        import base64
                        from io import BytesIO
                        header, encoded = img_path.split(",", 1)
                        img_data = base64.b64decode(encoded)
                        pil_images.append(self.Image.open(BytesIO(img_data)))
                    else:
                        # 本地文件路径
                        pil_images.append(self.Image.open(img_path))
                else:
                    # 假设已经是PIL Image
                    pil_images.append(img_path)
        
        # 用于保存输出信息（首次调用时使用）
        outputs_info = None
        inputs_info = None
        generation_method = None
        
        try:
            # 方法1: 使用processor（适用于视觉-语言模型）
            if self.has_processor and pil_images:
                generation_method = "processor_with_images"
                
                # BLIP VQA 的正确处理方式
                if self.model_class == 'blip_vqa':
                    # ✅ 正确：使用 BlipForQuestionAnswering
                    # processor 会正确处理 image 和 question
                    
                    # 标准调用方式
                    inputs = self.processor(
                        pil_images[0] if len(pil_images) == 1 else pil_images,  # 单图像传PIL.Image，多图像传list
                        prompt,  # question作为第二个参数
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # 首次调用时保存并打印输入信息
                    if self._first_call:
                        inputs_info = self._extract_inputs_info(inputs, pil_images)
                        self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
                        print("\n📥 处理后的输入结构 (BLIP VQA):")
                        self._print_inputs_structure(inputs_info)
                    
                    with self.torch.no_grad():
                        # BLIP VQA 推荐参数
                        generate_kwargs = {
                            "max_length": kwargs.get("max_length", 20),  # VQA答案通常很短
                        }
                        
                        # 不建议对VQA使用beam search和sampling，直接生成最可能的答案
                        # 但如果用户坚持，也可以添加
                        if "num_beams" in kwargs:
                            generate_kwargs["num_beams"] = kwargs["num_beams"]
                        
                        if self._first_call:
                            print(f"\n🔍 调试信息 - BLIP VQA Generate调用:")
                            print(f"  • 模型类: {type(self.model).__name__}")
                            print(f"  • inputs键: {list(inputs.keys())}")
                            print(f"  • generate_kwargs: {generate_kwargs}")
                        
                        # 标准VQA生成
                        outputs = self.model.generate(
                            **inputs,
                            **generate_kwargs
                        )
                        
                        if self._first_call:
                            print(f"  • outputs形状: {outputs.shape}")
                            print(f"  • outputs[0]: {outputs[0].tolist()}")
                    
                    # 首次调用时保存输出信息
                    if self._first_call:
                        outputs_info = self._extract_outputs_info(outputs, 0)
                    
                    # 直接解码答案
                    generated_text = self.processor.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    
                    if self._first_call:
                        print(f"  • 解码后的答案: '{generated_text}'")
                
                # BLIP Caption 模型（不是VQA）
                elif self.model_class == 'blip_caption':
                    # 对于Caption模型，同时处理图像和文本
                    inputs = self.processor(
                        images=pil_images,
                        text=prompt,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # 首次调用时保存并打印输入信息
                    if self._first_call:
                        inputs_info = self._extract_inputs_info(inputs, pil_images)
                        self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
                        print("\n📥 处理后的输入结构 (BLIP Caption):")
                        self._print_inputs_structure(inputs_info)
                    
                    # 记录问题的长度，用于后续截取答案
                    input_ids_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                    
                    with self.torch.no_grad():
                        # 生成参数
                        generate_kwargs = {
                            "max_length": kwargs.get("max_length", input_ids_length + 20),
                            "num_beams": kwargs.get("num_beams", 5),
                            "min_length": kwargs.get("min_length", 1),
                        }
                        
                        if self._first_call:
                            print(f"\n🔍 调试信息 - BLIP Caption Generate调用:")
                            print(f"  • input_ids_length: {input_ids_length}")
                            print(f"  • generate_kwargs: {generate_kwargs}")
                        
                        outputs = self.model.generate(
                            **inputs,
                            **generate_kwargs
                        )
                        
                        if self._first_call:
                            print(f"  • outputs形状: {outputs.shape}")
                            print(f"  • outputs[0]长度: {len(outputs[0])}")
                    
                    # 首次调用时保存输出信息
                    if self._first_call:
                        outputs_info = self._extract_outputs_info(outputs, input_ids_length)
                    
                    # 只解码新生成的部分
                    if len(outputs[0]) > input_ids_length:
                        answer_ids = outputs[0][input_ids_length:]
                        generated_text = self.processor.decode(
                            answer_ids,
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = self.processor.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                        if prompt.lower() in generated_text.lower():
                            generated_text = generated_text.lower().replace(prompt.lower(), "").strip()
                    
                    if self._first_call:
                        print(f"  • 解码后的文本: '{generated_text}'")
                else:
                    # 非BLIP模型，使用标准处理
                    inputs = self.processor(
                        text=prompt,
                        images=pil_images,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                             for k, v in inputs.items()}
                    input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                    
                    # 首次调用时保存并打印输入信息
                    if self._first_call:
                        inputs_info = self._extract_inputs_info(inputs, pil_images)
                        self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
                        print("\n📥 处理后的输入结构:")
                        self._print_inputs_structure(inputs_info)
                        print(f"  • 输入长度（用于解码）: {input_length}")
                    
                    with self.torch.no_grad():
                        # 生成参数
                        generate_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "do_sample": temperature > 0,
                        }
                        # 添加其他kwargs
                        for key, value in kwargs.items():
                            if key not in generate_kwargs and key not in ['max_tokens']:
                                generate_kwargs[key] = value
                        
                        outputs = self.model.generate(
                            **inputs,
                            **generate_kwargs
                        )
                    
                    # 首次调用时保存输出信息
                    if self._first_call:
                        outputs_info = self._extract_outputs_info(outputs, input_length)
                    
                    # 只解码新生成的部分
                    if input_length > 0 and len(outputs[0]) > input_length:
                        generated_ids = outputs[0][input_length:]
                        generated_text = self.processor.decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = self.processor.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                        if input_length > 0 and prompt in generated_text:
                            generated_text = generated_text.replace(prompt, "").strip()
            
            # 方法2: 使用chat接口（如果模型支持）
            elif hasattr(self.model, 'chat') and pil_images:
                generation_method = "chat_interface"
                # 首次调用时保存并打印输入信息
                if self._first_call:
                    inputs_info = {
                        "method": "chat",
                        "prompt": prompt,
                        "images_count": len(pil_images) if pil_images else 0,
                        "image_sizes": [img.size for img in pil_images] if pil_images else [],
                        "has_processor": self.has_processor
                    }
                    self._print_first_call_info(prompt, pil_images, max_new_tokens, temperature, top_p, kwargs)
                    print("\n📥 Chat接口输入信息:")
                    print(f"  • 图像数量: {inputs_info['images_count']}")
                    print(f"  • 图像尺寸: {inputs_info['image_sizes']}")
                    print(f"  • 使用Processor: {inputs_info['has_processor']}")
                
                if self.has_processor:
                    response, _ = self.model.chat(
                        self.processor,
                        query=prompt,
                        history=None,
                        images=pil_images,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    response, _ = self.model.chat(
                        self.tokenizer,
                        query=prompt,
                        history=None,
                        images=pil_images,
                    )
                generated_text = response
                
                # 首次调用时保存输出信息
                if self._first_call:
                    outputs_info = {
                        "method": "chat",
                        "response_type": type(response).__name__,
                        "response_length": len(response) if isinstance(response, str) else "N/A"
                    }
            
            # 方法3: 纯文本生成
            else:
                generation_method = "text_only"
                if self.has_processor:
                    inputs = self.processor(text=prompt, return_tensors="pt")
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                
                inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # 首次调用时保存并打印输入信息
                if self._first_call:
                    inputs_info = self._extract_inputs_info(inputs, None)
                    self._print_first_call_info(prompt, None, max_new_tokens, temperature, top_p, kwargs)
                    print("\n📥 处理后的输入结构:")
                    self._print_inputs_structure(inputs_info)
                
                # 保存输入长度
                input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        **kwargs
                    )
                
                # 首次调用时保存输出信息
                if self._first_call:
                    outputs_info = self._extract_outputs_info(outputs, input_length)
                
                # 只解码新生成的部分
                if input_length > 0 and len(outputs[0]) > input_length:
                    generated_ids = outputs[0][input_length:]
                    if self.has_processor:
                        generated_text = self.processor.decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = self.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                else:
                    if self.has_processor:
                        generated_text = self.processor.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = self.tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                    if prompt in generated_text:
                        generated_text = generated_text.replace(prompt, "").strip()
            
            result = {
                "text": generated_text,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "raw": {"generated_text": generated_text}
            }
            
            # 首次调用时输出详细信息
            if self._first_call:
                self._print_first_output_info(result, outputs_info, generation_method)
                self._first_call = False
                self._first_output_printed = True
            
            return result
        
        except Exception as e:
            raise RuntimeError(f"模型生成失败: {e}")
    
    def _print_inputs_structure(self, inputs_info: Dict):
        """打印输入结构信息"""
        if "keys" in inputs_info:
            print(f"  • 输入键: {inputs_info['keys']}")
        if "tensor_info" in inputs_info:
            print(f"  • Tensor信息:")
            for key, info in inputs_info["tensor_info"].items():
                if "shape" in info:
                    print(f"    - {key}: shape={info['shape']}, dtype={info['dtype']}, device={info['device']}")
                else:
                    print(f"    - {key}: {info}")
        if "images" in inputs_info:
            print(f"  • 图像信息:")
            for i, img_info in enumerate(inputs_info["images"]):
                print(f"    - 图像 {i+1}: {img_info}")
    
    def _print_first_call_info(self, prompt: str, images: Optional[List], max_new_tokens: int, 
                               temperature: float, top_p: float, kwargs: Dict):
        """打印首次调用时的详细信息"""
        print("\n" + "="*80)
        print("🔍 首次模型调用 - 输入信息")
        print("="*80)
        
        # 模型信息
        print("\n📦 模型信息:")
        print(f"  • 模型ID: {self.model_id}")
        print(f"  • 模型类型: {self.model_class}")
        print(f"  • 设备: {self.device}")
        print(f"  • 是否视觉模型: {self.is_vision_model}")
        print(f"  • 使用Processor: {self.has_processor}")
        if hasattr(self, 'model'):
            print(f"  • 模型类: {type(self.model).__name__}")
            if hasattr(self.model, 'config'):
                config = self.model.config
                print(f"  • 模型配置类型: {type(config).__name__}")
                if hasattr(config, 'vocab_size'):
                    print(f"  • 词汇表大小: {config.vocab_size}")
                if hasattr(config, 'max_position_embeddings'):
                    print(f"  • 最大位置编码: {config.max_position_embeddings}")
        
        # 输入信息
        print("\n📥 输入信息:")
        print(f"  • Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"  • Prompt长度: {len(prompt)} 字符")
        if images:
            print(f"  • 图像数量: {len(images)}")
            for i, img in enumerate(images):
                if isinstance(img, str):
                    print(f"    - 图像 {i+1}: {img[:100]}{'...' if len(img) > 100 else ''}")
                elif hasattr(img, 'size'):
                    print(f"    - 图像 {i+1}: PIL Image, 尺寸: {img.size}, 模式: {img.mode}")
                else:
                    print(f"    - 图像 {i+1}: {type(img).__name__}")
        else:
            print(f"  • 图像数量: 0")
        
        # 生成参数
        print("\n⚙️  生成参数:")
        print(f"  • max_new_tokens: {max_new_tokens}")
        print(f"  • temperature: {temperature}")
        print(f"  • top_p: {top_p}")
        if kwargs:
            print(f"  • 其他参数: {kwargs}")
        
        print("="*80 + "\n")
    
    def _extract_inputs_info(self, inputs: Dict, images: Optional[List]) -> Dict:
        """提取输入信息的结构化数据"""
        info = {
            "keys": list(inputs.keys()),
            "tensor_info": {}
        }
        
        for key, value in inputs.items():
            if isinstance(value, self.torch.Tensor):
                info["tensor_info"][key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "requires_grad": value.requires_grad
                }
            else:
                info["tensor_info"][key] = {
                    "type": type(value).__name__,
                    "value": str(value)[:100] if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
                }
        
        if images:
            info["images"] = []
            for img in images:
                if hasattr(img, 'size'):
                    info["images"].append({
                        "type": "PIL.Image",
                        "size": img.size,
                        "mode": img.mode
                    })
                else:
                    info["images"].append({"type": type(img).__name__})
        
        return info
    
    def _extract_outputs_info(self, outputs, input_length: int) -> Dict:
        """提取输出信息的结构化数据"""
        info = {
            "output_type": type(outputs).__name__,
            "input_length": input_length
        }
        
        if isinstance(outputs, self.torch.Tensor):
            info["shape"] = list(outputs.shape)
            info["dtype"] = str(outputs.dtype)
            info["device"] = str(outputs.device)
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            first_output = outputs[0] if isinstance(outputs, list) else outputs[0]
            if isinstance(first_output, self.torch.Tensor):
                info["first_output_shape"] = list(first_output.shape)
                info["first_output_dtype"] = str(first_output.dtype)
                info["total_length"] = first_output.shape[0] if len(first_output.shape) > 0 else "N/A"
                info["generated_length"] = first_output.shape[0] - input_length if input_length > 0 else "N/A"
        
        return info
    
    def _print_first_output_info(self, result: Dict, outputs_info: Optional[Dict], generation_method: Optional[str]):
        """打印首次调用时的输出信息"""
        print("\n" + "="*80)
        print("📤 首次模型调用 - 输出信息")
        print("="*80)
        
        # 生成方法
        if generation_method:
            print(f"\n🔧 使用的生成方法: {generation_method}")
        
        # 输出结构
        print("\n📊 输出结构:")
        print(f"  • 返回类型: {type(result).__name__}")
        print(f"  • 返回键: {list(result.keys())}")
        
        # 生成的文本
        if "text" in result:
            text = result["text"]
            print(f"\n💬 生成的文本:")
            print(f"  • 内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            print(f"  • 长度: {len(text)} 字符")
        
        # 原始输出信息
        if outputs_info:
            print(f"\n🔍 原始输出信息:")
            if "output_type" in outputs_info:
                print(f"  • 输出类型: {outputs_info['output_type']}")
            if "shape" in outputs_info:
                print(f"  • 形状: {outputs_info['shape']}")
            if "first_output_shape" in outputs_info:
                print(f"  • 第一个输出形状: {outputs_info['first_output_shape']}")
            if "input_length" in outputs_info:
                print(f"  • 输入长度: {outputs_info['input_length']}")
            if "generated_length" in outputs_info:
                print(f"  • 生成长度: {outputs_info['generated_length']}")
        
        # Usage信息
        if "usage" in result:
            print(f"\n📈 Token使用:")
            for key, value in result["usage"].items():
                print(f"  • {key}: {value}")
        
        # Raw信息（简要）
        if "raw" in result:
            raw = result["raw"]
            print(f"\n📦 Raw响应:")
            print(f"  • 类型: {type(raw).__name__}")
            if isinstance(raw, dict):
                print(f"  • 键: {list(raw.keys())}")
        
        print("="*80 + "\n")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_id,
            "type": "huggingface_local",
            "device": self.device,
            "model_class": self.model_class,
            "is_vision_model": self.is_vision_model
        }


# HuggingFace Hub Inference API适配器
class HuggingFaceHubAdapter(BaseModelAdapter):
    """HuggingFace Hub Inference API适配器"""
    
    def __init__(self, 
                 model_id: str,
                 api_token: Optional[str] = None,
                 api_url: Optional[str] = None,
                 timeout: float = 30.0,
                 **kwargs):
        """
        Args:
            model_id: HuggingFace模型ID
            api_token: HuggingFace API token（可选，用于私有模型）
            api_url: 自定义API URL（可选，默认使用HuggingFace Inference API）
            timeout: 请求超时时间
        """
        self.model_id = model_id
        self.api_token = api_token
        self.timeout = timeout
        
        if api_url:
            self.api_url = api_url.rstrip("/")
        else:
            self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("需要安装requests: pip install requests")
    
    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """生成模型响应"""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        # 构建请求payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 1024)),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }
        }
        
        # 处理图像
        if images:
            # HuggingFace Inference API支持图像输入
            # 对于视觉-语言模型，inputs可以是字典
            if len(images) == 1:
                img_path = images[0]
                if img_path.startswith("http") or img_path.startswith("data:image"):
                    payload["inputs"] = {
                        "text": prompt,
                        "image": img_path
                    }
                else:
                    # 读取本地图像并转换为base64
                    from PIL import Image
                    import base64
                    from io import BytesIO
                    
                    img = Image.open(img_path)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    payload["inputs"] = {
                        "text": prompt,
                        "image": f"data:image/jpeg;base64,{img_base64}"
                    }
        
        try:
            response = self.requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析响应（HuggingFace API返回格式可能不同）
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    text = result[0]["generated_text"]
                elif "answer" in result[0]:
                    text = result[0]["answer"]
                else:
                    text = str(result[0])
            elif isinstance(result, dict):
                text = result.get("generated_text", result.get("answer", str(result)))
            else:
                text = str(result)
            
            # 移除输入部分
            if prompt in text:
                text = text.replace(prompt, "").strip()
            
            return {
                "text": text,
                "usage": {},
                "raw": result
            }
        
        except Exception as e:
            raise RuntimeError(f"HuggingFace API调用失败: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_id,
            "type": "huggingface_hub_api",
            "api_url": self.api_url
        }


# 注册默认适配器
ModelAdapterFactory.register("openai", OpenAIAdapter)
ModelAdapterFactory.register("huggingface", HuggingFaceAdapter)
ModelAdapterFactory.register("hf", HuggingFaceAdapter)  # 简写
ModelAdapterFactory.register("huggingface_hub", HuggingFaceHubAdapter)
ModelAdapterFactory.register("hf_hub", HuggingFaceHubAdapter)  # 简写
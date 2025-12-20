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


# HuggingFace本地模型适配器
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
#             from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
#             from PIL import Image
#             import torch
#         except ImportError:
#             raise ImportError(
#                 "需要安装transformers和torch: pip install transformers torch pillow"
#             )
        
#         self.torch = torch
#         self.Image = Image
        
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
#                 dtype = self.torch.float16
#             else:
#                 dtype = self.torch.float32
#         elif dtype == "float16":
#             dtype = self.torch.float16
#         elif dtype == "bfloat16":
#             dtype = self.torch.bfloat16
#         else:
#             dtype = self.torch.float32
        
#         # 加载模型
#         load_kwargs = {
#             "trust_remote_code": trust_remote_code,
#             "device_map": device if device != "cpu" else None,
#         }
        
#         if device == "cpu":
#             load_kwargs["torch_dtype"] = dtype
#         else:
#             load_kwargs["torch_dtype"] = dtype
        
#         if load_in_8bit:
#             load_kwargs["load_in_8bit"] = True
#         elif load_in_4bit:
#             load_kwargs["load_in_4bit"] = True
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             **load_kwargs
#         )
#         self.model.eval()
    
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
        
#         try:
#             # 方法1: 使用processor（适用于视觉-语言模型）
#             if self.has_processor and pil_images:
#                 inputs = self.processor(
#                     text=prompt,
#                     images=pil_images,
#                     return_tensors="pt"
#                 )
#                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                          for k, v in inputs.items()}
                
#                 with self.torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=temperature > 0,
#                         **kwargs
#                     )
                
#                 # 解码输出
#                 generated_text = self.processor.decode(
#                     outputs[0],
#                     skip_special_tokens=True
#                 )
                
#                 # 移除输入部分
#                 if prompt in generated_text:
#                     generated_text = generated_text.replace(prompt, "").strip()
            
#             # 方法2: 使用chat接口（如果模型支持）
#             elif hasattr(self.model, 'chat') and pil_images:
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
            
#             # 方法3: 纯文本生成
#             else:
#                 if self.has_processor:
#                     inputs = self.processor(text=prompt, return_tensors="pt")
#                 else:
#                     inputs = self.tokenizer(prompt, return_tensors="pt")
                
#                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                          for k, v in inputs.items()}
                
#                 with self.torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=temperature > 0,
#                         **kwargs
#                     )
                
#                 if self.has_processor:
#                     generated_text = self.processor.decode(
#                         outputs[0],
#                         skip_special_tokens=True
#                     )
#                 else:
#                     generated_text = self.tokenizer.decode(
#                         outputs[0],
#                         skip_special_tokens=True
#                     )
                
#                 # 移除输入部分
#                 if prompt in generated_text:
#                     generated_text = generated_text.replace(prompt, "").strip()
            
#             return {
#                 "text": generated_text,
#                 "usage": {"prompt_tokens": 0, "completion_tokens": 0},  # 可以添加实际token统计
#                 "raw": {"generated_text": generated_text}
#             }
        
#         except Exception as e:
#             raise RuntimeError(f"模型生成失败: {e}")
    
#     def get_model_info(self) -> Dict[str, Any]:
#         return {
#             "name": self.model_id,
#             "type": "huggingface_local",
#             "device": self.device
#         }


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
#                 dtype = self.torch.float16
#             else:
#                 dtype = self.torch.float32
#         elif dtype == "float16":
#             dtype = self.torch.float16
#         elif dtype == "bfloat16":
#             dtype = self.torch.bfloat16
#         else:
#             dtype = self.torch.float32
        
#         # 准备加载参数
#         load_kwargs = {
#             "trust_remote_code": trust_remote_code,
#             "device_map": device if device != "cpu" else None,
#             "torch_dtype": dtype,
#         }
        
#         if load_in_8bit:
#             load_kwargs["load_in_8bit"] = True
#         elif load_in_4bit:
#             load_kwargs["load_in_4bit"] = True
        
#         # 根据模型类型选择合适的加载方式
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
#                 except:
#                     # 如果失败，尝试AutoModelForCausalLM（某些VL模型使用CausalLM架构）
#                     self.model = AutoModelForCausalLM.from_pretrained(
#                         model_id,
#                         **load_kwargs
#                     )
#                     self.model_class = 'causal_lm'
#             else:
#                 # 纯文本模型
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_id,
#                     **load_kwargs
#                 )
#                 self.model_class = 'causal_lm'
                
#         except Exception as e:
#             # 如果所有方法都失败，最后尝试AutoModelForCausalLM
#             print(f"警告: 使用标准方法加载失败，尝试备用方案: {e}")
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_id,
#                 **load_kwargs
#             )
#             self.model_class = 'causal_lm'
        
#         self.model.eval()
#         self.is_vision_model = is_vision_model
    
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
        
#         try:
#             # 方法1: 使用processor（适用于视觉-语言模型）
#             if self.has_processor and pil_images:
#                 inputs = self.processor(
#                     text=prompt,
#                     images=pil_images,
#                     return_tensors="pt"
#                 )
#                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                          for k, v in inputs.items()}
                
#                 with self.torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=temperature > 0,
#                         **kwargs
#                     )
                
#                 # 解码输出
#                 generated_text = self.processor.decode(
#                     outputs[0],
#                     skip_special_tokens=True
#                 )
                
#                 # 移除输入部分
#                 if prompt in generated_text:
#                     generated_text = generated_text.replace(prompt, "").strip()
            
#             # 方法2: 使用chat接口（如果模型支持）
#             elif hasattr(self.model, 'chat') and pil_images:
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
            
#             # 方法3: 纯文本生成
#             else:
#                 if self.has_processor:
#                     inputs = self.processor(text=prompt, return_tensors="pt")
#                 else:
#                     inputs = self.tokenizer(prompt, return_tensors="pt")
                
#                 inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
#                          for k, v in inputs.items()}
                
#                 with self.torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=temperature > 0,
#                         **kwargs
#                     )
                
#                 if self.has_processor:
#                     generated_text = self.processor.decode(
#                         outputs[0],
#                         skip_special_tokens=True
#                     )
#                 else:
#                     generated_text = self.tokenizer.decode(
#                         outputs[0],
#                         skip_special_tokens=True
#                     )
                
#                 # 移除输入部分
#                 if prompt in generated_text:
#                     generated_text = generated_text.replace(prompt, "").strip()
            
#             return {
#                 "text": generated_text,
#                 "usage": {"prompt_tokens": 0, "completion_tokens": 0},
#                 "raw": {"generated_text": generated_text}
#             }
        
#         except Exception as e:
#             raise RuntimeError(f"模型生成失败: {e}")
    
#     def get_model_info(self) -> Dict[str, Any]:
#         return {
#             "name": self.model_id,
#             "type": "huggingface_local",
#             "device": self.device,
#             "model_class": self.model_class,
#             "is_vision_model": self.is_vision_model
#         }


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
                # BLIP系列模型
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    **load_kwargs
                )
                self.model_class = 'blip'
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
        
        try:
            # 方法1: 使用processor（适用于视觉-语言模型）
            if self.has_processor and pil_images:
                inputs = self.processor(
                    text=prompt,
                    images=pil_images,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        **kwargs
                    )
                
                # 解码输出
                generated_text = self.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # 移除输入部分
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
            
            # 方法2: 使用chat接口（如果模型支持）
            elif hasattr(self.model, 'chat') and pil_images:
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
            
            # 方法3: 纯文本生成
            else:
                if self.has_processor:
                    inputs = self.processor(text=prompt, return_tensors="pt")
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                
                inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        **kwargs
                    )
                
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
                
                # 移除输入部分
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
            
            return {
                "text": generated_text,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "raw": {"generated_text": generated_text}
            }
        
        except Exception as e:
            raise RuntimeError(f"模型生成失败: {e}")
    
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

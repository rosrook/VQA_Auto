"""
Agent adapters: base interface + closed-source API adapter (e.g., OpenAI-compatible).

Design goals:
- 统一调用签名: generate(prompt, images=None, **kwargs) -> dict
- 清晰的错误/超时处理与可重试分类
- 便于扩展到其他服务 (Ollama / HF / Custom HTTP)
"""

import abc
import json
import time
from typing import Any, Dict, List, Optional

import requests


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class AgentError(Exception):
    """Base agent error."""


class AgentRetryableError(AgentError):
    """Transient error,可重试 (超时/限流/5xx)。"""


class AgentFatalError(AgentError):
    """不可重试错误 (鉴权/参数非法/配额耗尽)。"""


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class BaseAgent(abc.ABC):
    """统一的 Agent 抽象类."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        返回格式建议:
        {
            "text": "model output",
            "raw": {...},  # 原始响应
            "usage": {...} # token/price 等
        }
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI-compatible closed API adapter
# --------------------------------------------------------------------------- #
class OpenAIModelAgent(BaseAgent):
    """
    兼容 OpenAI / OpenAI-compatible API 的闭源模型适配器。
    仅实现文本/可选图像输入 (base64 或 url 由上层准备)。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    # 可重试
                    raise AgentRetryableError(f"status={resp.status_code}, body={resp.text}")
                if resp.status_code >= 400:
                    raise AgentFatalError(f"status={resp.status_code}, body={resp.text}")
                return resp.json()
            except AgentRetryableError as e:
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(1.5 * attempt)  # 简单退避
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    raise AgentRetryableError(f"request failed after retries: {e}") from e
                time.sleep(1.5 * attempt)
            except Exception as e:
                raise AgentFatalError(f"unexpected error: {e}") from e
        # 理论不会到这里
        raise AgentRetryableError(f"unresolved error: {last_err}")

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        images: 可选的已处理图像输入（如 base64 或 URL），按需在 messages 中组织。
        此处示例为纯文本；多模态可在 content 中插入 image_url / image_base64。
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        raw = self._request(payload)
        choice = raw.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = raw.get("usage", {})

        return {
            "text": text,
            "raw": raw,
            "usage": usage,
        }


# --------------------------------------------------------------------------- #
# Local HF / Transformers adapter
# --------------------------------------------------------------------------- #
class LocalHFModelAgent(BaseAgent):
    """
    本地 Transformers 推理适配器 (文本/可选图像占位).
    依赖: transformers >= 4.38 (建议), torch
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device,
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        images: 目前未内置处理，如需多模态需自行扩展/换用多模态模型与对应管线。
        """
        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "do_sample": self.do_sample,
        }
        if stop:
            gen_kwargs["eos_token_id"] = stop  # 简化：调用方可传单个或列表，需要 tokenizer 兼容

        outputs = self.pipe(prompt, **gen_kwargs)
        # transformers pipeline 返回 list[dict]
        text = outputs[0]["generated_text"]

        return {
            "text": text,
            "raw": outputs,
            "usage": {},
        }


# --------------------------------------------------------------------------- #
# Remote HF Hub Inference adapter (text-only, OpenAI-like endpoint)
# --------------------------------------------------------------------------- #
class RemoteHFModelAgent(BaseAgent):
    """
    远端 HuggingFace Hub Inference / Text Generation Inference 适配器
    (假设兼容 OpenAI completion/chat 风格的 HTTP 接口).
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.endpoint}/chat/completions"

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise AgentRetryableError(f"status={resp.status_code}, body={resp.text}")
                if resp.status_code >= 400:
                    raise AgentFatalError(f"status={resp.status_code}, body={resp.text}")
                return resp.json()
            except AgentRetryableError as e:
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(1.5 * attempt)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    raise AgentRetryableError(f"request failed after retries: {e}") from e
                time.sleep(1.5 * attempt)
            except Exception as e:
                raise AgentFatalError(f"unexpected error: {e}") from e
        raise AgentRetryableError(f"unresolved error: {last_err}")

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        raw = self._request(payload)
        choice = raw.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = raw.get("usage", {})

        return {
            "text": text,
            "raw": raw,
            "usage": usage,
        }


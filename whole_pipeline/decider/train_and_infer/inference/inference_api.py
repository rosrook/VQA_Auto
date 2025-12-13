"""
模型推理接口：通过版本号加载和运行模型
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
import torch

import sys
from pathlib import Path
# 添加父目录到路径，以便导入 version 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from version.version_manager import VersionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """模型推理类"""
    
    def __init__(
        self,
        version: Optional[str] = None,
        model_path: Optional[str] = None,
        registry_file: str = "model_registry.json",
        base_dir: str = "./model_versions",
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        初始化推理接口
        
        参数:
            version: 模型版本号（如果提供，从注册表加载）
            model_path: 直接指定模型路径（优先级高于 version）
            registry_file: 版本注册表文件路径
            base_dir: 模型版本存储目录
            device: 设备（"auto", "cuda", "cpu"）
            torch_dtype: 数据类型（"auto", "float16", "bfloat16", "float32"）
        """
        self.manager = VersionManager(registry_file=registry_file, base_dir=base_dir)
        
        # 确定模型路径
        if model_path:
            self.model_path = Path(model_path)
        elif version:
            model_path = self.manager.get_model_path(version)
            if not model_path:
                raise ValueError(f"版本 {version} 不存在或模型路径无效")
            self.model_path = Path(model_path)
            logger.info(f"从版本 {version} 加载模型: {self.model_path}")
        else:
            # 使用最新版本
            latest_version = self.manager.get_latest_version()
            if not latest_version:
                raise ValueError("没有可用的模型版本")
            model_path = self.manager.get_model_path(latest_version)
            self.model_path = Path(model_path)
            logger.info(f"使用最新版本 {latest_version}: {self.model_path}")
        
        if not self.model_path.exists():
            raise ValueError(f"模型路径不存在: {self.model_path}")
        
        # 确定设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 确定数据类型
        if torch_dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, torch_dtype)
        
        # 加载模型和处理器
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        model_path_str = str(self.model_path)
        
        logger.info(f"加载模型: {model_path_str}")
        logger.info(f"设备: {self.device}, 数据类型: {self.torch_dtype}")
        
        # 判断模型类型
        if "Qwen2-VL" in model_path_str or "Qwen2VL" in model_path_str:
            # Qwen2-VL 模型
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path_str,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.processor = AutoProcessor.from_pretrained(model_path_str)
            self.is_multimodal = True
        else:
            # 标准语言模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.processor = AutoTokenizer.from_pretrained(model_path_str)
            self.is_multimodal = False
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("✓ 模型加载完成")
    
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        images: Optional[List] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        生成文本
        
        参数:
            prompt: 输入提示（字符串或对话格式）
            images: 图像列表（仅多模态模型）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否采样
            **kwargs: 其他生成参数
            
        返回:
            生成的文本
        """
        # 准备输入
        if self.is_multimodal:
            # 多模态输入
            if isinstance(prompt, str):
                # 转换为对话格式
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            # 处理输入
            text_inputs = self.processor(
                text=messages,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        else:
            # 文本输入
            if isinstance(prompt, list):
                # 对话格式，需要应用模板
                from trl.data_utils import apply_chat_template
                text = apply_chat_template({"messages": prompt}, self.processor)["text"]
            else:
                text = prompt
            
            text_inputs = self.processor(
                text,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **kwargs
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **text_inputs,
                generation_config=generation_config,
            )
        
        # 解码
        if self.is_multimodal:
            generated_text = self.processor.batch_decode(
                outputs[:, text_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
        else:
            generated_text = self.processor.decode(
                outputs[0][text_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        
        return generated_text
    
    def predict_agent_selection(
        self,
        report: str,
        available_agents: List[str],
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        预测 agent 选择（针对数据过滤任务）
        
        参数:
            report: 数据集报告
            available_agents: 可用的 agent 列表
            system_prompt: 系统提示（如果为 None，使用默认）
            **generation_kwargs: 生成参数
            
        返回:
            解析后的结果字典
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
        from grpo import create_system_prompt, parse_agent_selection
        
        # 创建系统提示
        if system_prompt is None:
            system_prompt = create_system_prompt(available_agents)
        
        # 构建对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"REPORT IS :\n {report}"},
        ]
        
        # 生成
        generated_text = self.generate(messages, **generation_kwargs)
        
        # 解析结果
        result = parse_agent_selection(generated_text)
        result["raw_output"] = generated_text
        
        return result


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="模型推理接口")
    parser.add_argument("--version", type=str, help="模型版本号")
    parser.add_argument("--model-path", type=str, help="直接指定模型路径")
    parser.add_argument("--registry-file", type=str, default="model_registry.json",
                       help="版本注册表文件路径")
    parser.add_argument("--base-dir", type=str, default="./model_versions",
                       help="模型版本存储目录")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="设备")
    parser.add_argument("--torch-dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="数据类型")
    
    # 推理参数
    parser.add_argument("--prompt", type=str, help="输入提示")
    parser.add_argument("--report", type=str, help="数据集报告（用于agent选择任务）")
    parser.add_argument("--available-agents", type=str, nargs="+",
                       help="可用的agent列表")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    
    # 交互模式
    parser.add_argument("--interactive", action="store_true",
                       help="交互模式")
    
    args = parser.parse_args()
    
    # 初始化推理接口
    try:
        inference = ModelInference(
            version=args.version,
            model_path=args.model_path,
            registry_file=args.registry_file,
            base_dir=args.base_dir,
            device=args.device,
            torch_dtype=args.torch_dtype
        )
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return 1
    
    # 交互模式
    if args.interactive:
        print("=" * 60)
        print("交互式推理模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n输入提示: ").strip()
                if prompt.lower() in ["quit", "exit"]:
                    break
                
                if not prompt:
                    continue
                
                result = inference.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                print(f"\n生成结果:\n{result}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"生成失败: {e}")
        
        return 0
    
    # 单次推理
    if args.report and args.available_agents:
        # Agent 选择任务
        result = inference.predict_agent_selection(
            report=args.report,
            available_agents=args.available_agents,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.prompt:
        # 普通文本生成
        result = inference.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(result)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


"""
版本管理系统：管理训练模型的版本和元数据
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    model_path: str
    config_path: Optional[str] = None
    created_at: str = None
    training_epochs: Optional[int] = None
    final_reward: Optional[float] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class VersionManager:
    """版本管理器"""
    
    def __init__(self, registry_file: str = "model_registry.json", base_dir: str = "./model_versions"):
        """
        初始化版本管理器
        
        参数:
            registry_file: 版本注册表文件路径
            base_dir: 模型版本存储的基础目录
        """
        self.registry_file = Path(registry_file)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载注册表
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """加载版本注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载注册表失败: {e}，创建新注册表")
        return {}
    
    def _save_registry(self):
        """保存版本注册表"""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
        logger.info(f"版本注册表已保存: {self.registry_file}")
    
    def _generate_version(self, prefix: str = "v") -> str:
        """
        生成新版本号
        
        格式: v{YYYYMMDD}_{序号}
        例如: v20241212_001
        """
        today = datetime.now().strftime("%Y%m%d")
        
        # 查找今天的版本
        today_versions = [
            v for v in self.registry.keys()
            if v.startswith(f"{prefix}{today}_")
        ]
        
        if today_versions:
            # 获取最大序号
            max_num = max([
                int(v.split("_")[-1]) for v in today_versions
            ])
            next_num = max_num + 1
        else:
            next_num = 1
        
        version = f"{prefix}{today}_{next_num:03d}"
        return version
    
    def register_version(
        self,
        model_path: str,
        version: Optional[str] = None,
        config_path: Optional[str] = None,
        training_epochs: Optional[int] = None,
        final_reward: Optional[float] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        auto_version: bool = True
    ) -> str:
        """
        注册新版本
        
        参数:
            model_path: 模型路径（训练输出目录）
            version: 版本号（如果为 None，自动生成）
            config_path: 配置文件路径
            training_epochs: 训练轮数
            final_reward: 最终奖励值
            description: 版本描述
            metadata: 额外元数据
            auto_version: 是否自动生成版本号
            
        返回:
            版本号
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"模型路径不存在: {model_path}")
        
        # 生成版本号
        if version is None and auto_version:
            version = self._generate_version()
        elif version is None:
            raise ValueError("必须提供版本号或启用 auto_version")
        
        # 检查版本是否已存在
        if version in self.registry:
            raise ValueError(f"版本 {version} 已存在")
        
        # 创建版本目录
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        logger.info(f"复制模型文件到版本目录: {version_dir}")
        if model_path.is_dir():
            # 如果是目录，复制整个目录
            dest_model_path = version_dir / "model"
            if dest_model_path.exists():
                shutil.rmtree(dest_model_path)
            shutil.copytree(model_path, dest_model_path)
            model_path_str = str(dest_model_path)
        else:
            # 如果是文件，复制文件
            dest_model_path = version_dir / model_path.name
            shutil.copy2(model_path, dest_model_path)
            model_path_str = str(dest_model_path)
        
        # 复制配置文件（如果提供）
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                dest_config_path = version_dir / "config.yaml"
                shutil.copy2(config_path, dest_config_path)
                config_path_str = str(dest_config_path)
            else:
                logger.warning(f"配置文件不存在: {config_path}")
                config_path_str = None
        else:
            config_path_str = None
        
        # 创建版本信息
        version_info = ModelVersion(
            version=version,
            model_path=model_path_str,
            config_path=config_path_str,
            training_epochs=training_epochs,
            final_reward=final_reward,
            description=description,
            metadata=metadata or {}
        )
        
        # 保存版本信息到注册表
        self.registry[version] = asdict(version_info)
        self._save_registry()
        
        # 保存版本信息到版本目录
        version_info_file = version_dir / "version_info.json"
        with open(version_info_file, "w", encoding="utf-8") as f:
            json.dump(asdict(version_info), f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 版本 {version} 注册成功")
        logger.info(f"  模型路径: {model_path_str}")
        logger.info(f"  版本目录: {version_dir}")
        
        return version
    
    def register_base_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        """
        注册基础模型（从 HuggingFace 拉取后自动注册为初版）
        
        参数:
            model_id: HuggingFace 模型 ID（例如: "Qwen/Qwen2-VL-7B-Instruct"）
            version: 版本号（如果为 None，使用 "v0_base" 或自动生成）
            description: 版本描述（默认: "基础模型 - {model_id}"）
            cache_dir: HuggingFace 缓存目录（如果为 None，使用默认缓存）
            
        返回:
            版本号
        """
        from huggingface_hub import snapshot_download
        
        # 确定版本号
        if version is None:
            # 检查是否已有基础版本
            base_versions = [v for v in self.registry.keys() if v.startswith("v0_base")]
            if base_versions:
                # 如果已有基础版本，生成新的基础版本号
                max_num = max([int(v.split("_")[-1]) if v.split("_")[-1].isdigit() else 0 
                              for v in base_versions])
                version = f"v0_base_{max_num + 1:03d}"
            else:
                version = "v0_base"
        
        # 检查版本是否已存在
        if version in self.registry:
            logger.warning(f"版本 {version} 已存在，跳过注册")
            return version
        
        # 确定缓存目录
        if cache_dir is None:
            # 尝试从环境变量获取，否则使用默认路径
            import os
            cache_dir = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub"))
        cache_dir = Path(cache_dir)
        
        # 从 HuggingFace 下载模型
        logger.info(f"正在从 HuggingFace 拉取模型: {model_id}")
        try:
            # 使用 snapshot_download 下载完整模型
            model_cache_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
            )
            logger.info(f"模型已下载到: {model_cache_path}")
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            raise
        
        # 创建版本目录
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件到版本目录（创建符号链接以节省空间）
        logger.info(f"复制模型文件到版本目录: {version_dir}")
        dest_model_path = version_dir / "model"
        if dest_model_path.exists():
            shutil.rmtree(dest_model_path)
        
        # 复制整个模型目录
        shutil.copytree(model_cache_path, dest_model_path)
        model_path_str = str(dest_model_path)
        
        # 创建版本信息
        if description is None:
            description = f"基础模型 - {model_id}"
        
        version_info = ModelVersion(
            version=version,
            model_path=model_path_str,
            config_path=None,
            training_epochs=None,
            final_reward=None,
            description=description,
            metadata={
                "model_id": model_id,
                "is_base_model": True,
                "source": "huggingface",
                "cache_path": str(model_cache_path),
            }
        )
        
        # 保存版本信息到注册表
        self.registry[version] = asdict(version_info)
        self._save_registry()
        
        # 保存版本信息到版本目录
        version_info_file = version_dir / "version_info.json"
        with open(version_info_file, "w", encoding="utf-8") as f:
            json.dump(asdict(version_info), f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 基础模型已注册为版本: {version}")
        logger.info(f"  模型路径: {model_path_str}")
        logger.info(f"  版本目录: {version_dir}")
        
        return version
    
    def get_version(self, version: str) -> Optional[Dict]:
        """获取版本信息"""
        return self.registry.get(version)
    
    def list_versions(self) -> List[str]:
        """列出所有版本"""
        return sorted(self.registry.keys(), reverse=True)
    
    def get_latest_version(self) -> Optional[str]:
        """获取最新版本"""
        versions = self.list_versions()
        return versions[0] if versions else None
    
    def get_model_path(self, version: str) -> Optional[str]:
        """获取指定版本的模型路径"""
        version_info = self.get_version(version)
        if version_info:
            return version_info.get("model_path")
        return None
    
    def delete_version(self, version: str):
        """删除版本"""
        if version not in self.registry:
            raise ValueError(f"版本 {version} 不存在")
        
        # 删除版本目录
        version_dir = self.base_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            logger.info(f"已删除版本目录: {version_dir}")
        
        # 从注册表移除
        del self.registry[version]
        self._save_registry()
        
        logger.info(f"✓ 版本 {version} 已删除")


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型版本管理工具")
    parser.add_argument("command", choices=["register", "register-base", "list", "get", "delete", "latest"],
                       help="命令")
    parser.add_argument("--version", type=str, help="版本号")
    parser.add_argument("--model-path", type=str, help="模型路径")
    parser.add_argument("--model-id", type=str, help="HuggingFace 模型 ID（用于 register-base）")
    parser.add_argument("--config-path", type=str, help="配置文件路径")
    parser.add_argument("--description", type=str, help="版本描述")
    parser.add_argument("--cache-dir", type=str, help="HuggingFace 缓存目录")
    parser.add_argument("--registry-file", type=str, default="model_registry.json",
                       help="注册表文件路径")
    parser.add_argument("--base-dir", type=str, default="./model_versions",
                       help="模型版本存储目录")
    
    args = parser.parse_args()
    
    manager = VersionManager(
        registry_file=args.registry_file,
        base_dir=args.base_dir
    )
    
    if args.command == "register":
        if not args.model_path:
            print("错误: --model-path 是必需的")
            return 1
        
        version = manager.register_version(
            model_path=args.model_path,
            version=args.version,
            config_path=args.config_path,
            description=args.description
        )
        print(f"✓ 版本 {version} 注册成功")
    
    elif args.command == "register-base":
        if not args.model_id:
            print("错误: --model-id 是必需的（例如: Qwen/Qwen2-VL-7B-Instruct）")
            return 1
        
        version = manager.register_base_model(
            model_id=args.model_id,
            version=args.version,
            description=args.description,
            cache_dir=args.cache_dir
        )
        print(f"✓ 基础模型已注册为版本: {version}")
    
    elif args.command == "list":
        versions = manager.list_versions()
        if versions:
            print("可用版本:")
            for v in versions:
                info = manager.get_version(v)
                print(f"  {v}: {info.get('created_at', 'N/A')} - {info.get('description', 'N/A')}")
        else:
            print("没有注册的版本")
    
    elif args.command == "get":
        if not args.version:
            print("错误: --version 是必需的")
            return 1
        
        info = manager.get_version(args.version)
        if info:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(f"版本 {args.version} 不存在")
            return 1
    
    elif args.command == "latest":
        latest = manager.get_latest_version()
        if latest:
            print(latest)
        else:
            print("没有注册的版本")
            return 1
    
    elif args.command == "delete":
        if not args.version:
            print("错误: --version 是必需的")
            return 1
        
        manager.delete_version(args.version)
        print(f"✓ 版本 {args.version} 已删除")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


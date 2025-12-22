#!/usr/bin/env python3
"""
测试脚本：验证所有模块的导入路径是否正确
"""
import sys
from pathlib import Path

# 确保src目录在路径中
_script_dir = Path(__file__).parent.absolute()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

print("=" * 60)
print("测试模块导入")
print("=" * 60)

# 测试data模块
try:
    from data.data_pipeline import DataPipeline
    print("✓ data.data_pipeline")
except Exception as e:
    print(f"✗ data.data_pipeline: {e}")

try:
    from data.data_loader import DataLoader
    print("✓ data.data_loader")
except Exception as e:
    print(f"✗ data.data_loader: {e}")

try:
    from data.data_processor import DataProcessor
    print("✓ data.data_processor")
except Exception as e:
    print(f"✗ data.data_processor: {e}")

# 测试models模块
try:
    from models.model_loader import ModelLoader
    print("✓ models.model_loader")
except Exception as e:
    print(f"✗ models.model_loader: {e}")

try:
    from models.model_utils import count_parameters
    print("✓ models.model_utils")
except Exception as e:
    print(f"✗ models.model_utils: {e}")

# 测试training模块
try:
    from training.trainer import Trainer
    print("✓ training.trainer")
except Exception as e:
    print(f"✗ training.trainer: {e}")

try:
    from training.callbacks import EarlyStoppingCallback
    print("✓ training.callbacks")
except Exception as e:
    print(f"✗ training.callbacks: {e}")

try:
    from training.evaluator import Evaluator
    print("✓ training.evaluator")
except Exception as e:
    print(f"✗ training.evaluator: {e}")

# 测试utils模块
try:
    from utils.logger import Logger
    print("✓ utils.logger")
except Exception as e:
    print(f"✗ utils.logger: {e}")

try:
    from utils.checkpoint import CheckpointManager
    print("✓ utils.checkpoint")
except Exception as e:
    print(f"✗ utils.checkpoint: {e}")

try:
    from utils.metrics import accuracy
    print("✓ utils.metrics")
except Exception as e:
    print(f"✗ utils.metrics: {e}")

# 测试pipeline
try:
    from pipeline import VQAPipeline
    print("✓ pipeline")
except Exception as e:
    print(f"✗ pipeline: {e}")

print("=" * 60)
print("导入测试完成")
print("=" * 60)


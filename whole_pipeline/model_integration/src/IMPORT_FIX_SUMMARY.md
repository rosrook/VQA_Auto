# 导入路径修复总结

## 修复内容

已统一修复所有跨文件引用问题，确保可以直接运行 `pipeline.py`。

### 1. 统一导入方式

所有模块现在使用**绝对导入**（相对于 `src` 目录）：

- ✅ `from data.data_pipeline import DataPipeline`
- ✅ `from models.model_loader import ModelLoader`
- ✅ `from training.trainer import Trainer`
- ✅ `from utils.logger import Logger`

### 2. 修复的文件

#### 核心模块
- ✅ `pipeline.py` - 添加路径处理，确保可以直接运行
- ✅ `data/data_pipeline.py` - 修复所有内部导入
- ✅ `training/trainer.py` - 移除 `sys.path` 操作，使用绝对导入

#### 初始化文件
- ✅ `data/__init__.py` - 创建并修复导入
- ✅ `models/__init__.py` - 修复导入
- ✅ `training/__init__.py` - 修复导入
- ✅ `utils/__init__.py` - 修复导入

#### 其他文件
- ✅ `data/memory_utils.py` - 修复导入
- ✅ `data/dataset_optimized.py` - 修复导入
- ✅ `data/example_create_dataloader.py` - 修复导入

## 使用方法

### 方法1：直接运行 pipeline.py（推荐）

```bash
cd /path/to/model_integration/src
python pipeline.py --config config/vqa_config.yaml
```

`pipeline.py` 会自动处理路径，确保所有模块可以正确导入。

### 方法2：作为模块导入

```python
# 确保 src 目录在 Python 路径中
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/path/to/model_integration/src')))

# 然后导入
from pipeline import VQAPipeline
from data.data_pipeline import DataPipeline
from models.model_loader import ModelLoader
```

### 方法3：使用包结构

如果从 `model_integration` 目录运行：

```bash
cd /path/to/model_integration
python -m src.pipeline --config config/vqa_config.yaml
```

## 导入规则

### ✅ 正确的导入方式

```python
# 跨模块导入（使用绝对导入）
from data.data_pipeline import DataPipeline
from models.model_loader import ModelLoader
from training.trainer import Trainer
from utils.logger import Logger

# 同一模块内导入（可以使用相对导入）
from .dataset import VQADataset  # 在 data 模块内
```

### ❌ 错误的导入方式

```python
# 不要使用相对导入跨模块
from data_loader import DataLoader  # ❌
from ..data.data_loader import DataLoader  # ❌（在顶层）

# 不要手动修改 sys.path（除非必要）
sys.path.append(...)  # ❌（pipeline.py 已处理）
```

## 测试导入

运行测试脚本验证导入是否正确：

```bash
cd /path/to/model_integration/src
python test_imports.py
```

## 注意事项

1. **路径处理**：`pipeline.py` 会自动将 `src` 目录添加到 Python 路径，确保可以直接运行
2. **包结构**：所有模块都使用 `data.xxx`、`models.xxx` 等绝对导入
3. **向后兼容**：修复后的代码保持了原有的功能，只是统一了导入方式

## 目录结构

```
src/
├── pipeline.py          # 主入口（可直接运行）
├── data/
│   ├── __init__.py      # 导出主要类
│   ├── data_pipeline.py
│   ├── data_loader.py
│   └── ...
├── models/
│   ├── __init__.py
│   ├── model_loader.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── ...
└── utils/
    ├── __init__.py
    ├── logger.py
    └── ...
```

## 常见问题

### Q: 运行 pipeline.py 时提示 "ModuleNotFoundError"

**A:** 确保：
1. 在 `src` 目录下运行：`cd src && python pipeline.py`
2. 或者使用模块方式：`python -m src.pipeline`（从 `model_integration` 目录）

### Q: 导入时提示循环导入

**A:** 检查是否有模块互相导入。如果存在，考虑：
1. 将共同依赖提取到独立模块
2. 使用延迟导入（在函数内部导入）

### Q: 如何添加新模块？

**A:** 
1. 在相应目录下创建文件（如 `data/new_module.py`）
2. 使用绝对导入：`from data.new_module import ...`
3. 在 `__init__.py` 中导出（可选）


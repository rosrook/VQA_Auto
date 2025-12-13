# 版本管理和推理接口使用指南

## 概述

系统提供了完整的版本管理功能，支持：
- **基础模型注册**: 从 HuggingFace 拉取模型后自动注册为初版
- **未训练模型推理**: 支持对未训练的基础模型进行推理
- **训练版本管理**: 训练完成后自动保存模型版本
- **版本调用**: 通过版本号随时调用特定版本的模型进行推理

## 功能特性

1. **基础模型注册**: 从 HuggingFace 拉取模型后自动注册为初版（例如：`v0_base`）
2. **未训练模型推理**: 支持对未训练的基础模型进行推理
3. **自动版本管理**: 训练完成后自动注册模型版本
4. **版本查询**: 查看所有可用版本
5. **版本调用**: 通过版本号加载特定版本的模型
6. **推理接口**: 提供便捷的推理API

## 工作流程

### 0. 拉取并注册基础模型（首次使用）

在开始训练之前，您需要先从 HuggingFace 拉取模型并注册为初版：

#### 方式 1: 使用脚本（推荐）

```bash
# 从 HuggingFace 拉取模型并自动注册
python scripts/download_and_register_model.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct

# 指定版本号和描述
python scripts/download_and_register_model.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --version v0_base_qwen2vl \
    --description "Qwen2-VL 7B 基础模型"
```

#### 方式 2: 使用版本管理工具

```bash
python version/version_manager.py register-base \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --version v0_base \
    --description "基础模型 - Qwen2-VL-7B"
```

拉取完成后，会自动：
- 从 HuggingFace 下载模型到缓存目录
- 复制模型到 `./model_versions/v0_base/model/`
- 注册版本到 `model_registry.json`
- 标记为未训练的基础模型

#### 使用未训练模型进行推理

拉取模型后，可以直接使用未训练的模型进行推理：

```bash
python inference/run_inference.py \
    --version v0_base \
    --report "Dataset contains 100 samples with 15% duplication rate..." \
    --available-agents quality_filter_agent deduplication_filter
```

### 1. 训练模型（自动注册版本）

运行训练时，模型会自动注册版本：

```bash
python grpo.py --config test_config.yaml
```

训练完成后，会自动：
- 保存模型到 `./test_output/`
- 注册版本到 `model_registry.json`
- 复制模型到 `./model_versions/v{YYYYMMDD}_{序号}/`

### 2. 查看可用版本

```bash
# 列出所有版本
python version_manager.py list

# 查看特定版本信息
python version_manager.py get --version v20241212_001

# 获取最新版本号
python version_manager.py latest
```

### 3. 使用模型进行推理

#### 方式 1: 使用推理脚本（推荐）

```bash
python run_inference.py \
    --version v20241212_001 \
    --report "Dataset contains 100 samples with 15% duplication rate..." \
    --available-agents quality_filter_agent deduplication_filter length_filter
```

#### 方式 2: 使用推理API（Python）

```python
from inference_api import ModelInference

# 初始化（使用版本号）
inference = ModelInference(version="v20241212_001")

# 或使用最新版本
inference = ModelInference()  # 自动使用最新版本

# 预测 agent 选择
result = inference.predict_agent_selection(
    report="数据集报告文本...",
    available_agents=["quality_filter_agent", "deduplication_filter"]
)

print(result)
# {
#     "agents": ["quality_filter_agent", "deduplication_filter"],
#     "prompts": {
#         "quality_filter_agent": "prompt text...",
#         "deduplication_filter": "prompt text..."
#     },
#     "reasoning": "...",
#     "raw_output": "..."
# }
```

#### 方式 3: 交互式推理

```bash
python inference_api.py \
    --version v20241212_001 \
    --interactive
```

## 版本管理命令

### 注册版本（手动）

#### 注册训练后的模型版本

如果需要手动注册训练后的模型版本：

```bash
python version/version_manager.py register \
    --model-path ./test_output \
    --config-path test_config.yaml \
    --description "测试训练 - 第1次"
```

#### 注册基础模型版本

如果需要手动注册基础模型（从 HuggingFace 拉取）：

```bash
python version/version_manager.py register-base \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --version v0_base \
    --description "基础模型 - Qwen2-VL-7B"
```

### 查看版本列表

```bash
python version_manager.py list
```

输出示例：
```
可用版本:
  v20241212_003: 2024-12-12T20:30:00 - 测试训练 - 第3次
  v20241212_002: 2024-12-12T19:15:00 - 测试训练 - 第2次
  v20241212_001: 2024-12-12T18:00:00 - 测试训练 - 第1次
```

### 获取版本信息

```bash
python version_manager.py get --version v20241212_001
```

### 删除版本

```bash
python version_manager.py delete --version v20241212_001
```

## 目录结构

```
decider/
├── model_registry.json          # 版本注册表
├── model_versions/              # 版本存储目录
│   ├── v20241212_001/
│   │   ├── model/               # 模型文件
│   │   ├── config.yaml          # 配置文件（如果提供）
│   │   └── version_info.json    # 版本信息
│   ├── v20241212_002/
│   │   └── ...
│   └── ...
├── version_manager.py           # 版本管理工具
├── inference_api.py             # 推理API
└── run_inference.py             # 简化推理脚本
```

## 版本号格式

### 训练版本格式

训练后的版本号格式：`v{YYYYMMDD}_{序号}`

- `v`: 版本前缀
- `YYYYMMDD`: 日期（年-月-日）
- `序号`: 当天的序号（001, 002, ...）

示例：
- `v20241212_001` - 2024年12月12日的第1个训练版本
- `v20241212_002` - 2024年12月12日的第2个训练版本
- `v20241213_001` - 2024年12月13日的第1个训练版本

### 基础模型版本格式

基础模型（未训练）的版本号格式：`v0_base` 或 `v0_base_{序号}`

- `v0_base` - 第一个基础模型版本
- `v0_base_001` - 第二个基础模型版本
- `v0_base_qwen2vl` - 自定义名称的基础模型版本

示例：
- `v0_base` - 默认的基础模型版本
- `v0_base_001` - 第二个基础模型版本
- `v0_base_qwen2vl` - Qwen2-VL 基础模型

## 推理接口参数

### ModelInference 初始化参数

- `version`: 模型版本号（可选）
- `model_path`: 直接指定模型路径（可选，优先级高于version）
- `registry_file`: 注册表文件路径（默认: "model_registry.json"）
- `base_dir`: 版本存储目录（默认: "./model_versions"）
- `device`: 设备（"auto", "cuda", "cpu"）
- `torch_dtype`: 数据类型（"auto", "float16", "bfloat16", "float32"）

### predict_agent_selection 参数

- `report`: 数据集报告文本
- `available_agents`: 可用的agent列表
- `system_prompt`: 系统提示（可选，默认自动生成）
- `max_new_tokens`: 最大生成token数（默认: 512）
- `temperature`: 温度参数（默认: 0.7）
- `top_p`: top-p采样（默认: 0.9）
- `do_sample`: 是否采样（默认: True）

## 使用示例

### 示例 1: 训练并自动注册版本

```bash
# 训练模型
python grpo.py --config test_config.yaml

# 训练完成后会自动注册版本，输出类似：
# ✓ 模型已注册为版本: v20241212_001
```

### 示例 2: 使用特定版本进行推理

```bash
# 使用版本号
python run_inference.py \
    --version v20241212_001 \
    --report "Dataset contains 100 samples..." \
    --available-agents quality_filter_agent deduplication_filter

# 使用最新版本（不指定 --version）
python run_inference.py \
    --report "Dataset contains 100 samples..." \
    --available-agents quality_filter_agent deduplication_filter
```

### 示例 3: Python API 使用

```python
from inference_api import ModelInference

# 初始化
inference = ModelInference(version="v20241212_001")

# 准备输入
report = """
Dataset contains 10000 samples with high duplication rate (35%).
Quality issues: 15% samples have incomplete information.
Average text length: 250 tokens.
"""

available_agents = [
    "quality_filter_agent",
    "deduplication_filter",
    "length_filter"
]

# 生成预测
result = inference.predict_agent_selection(
    report=report,
    available_agents=available_agents,
    max_new_tokens=512,
    temperature=0.7
)

# 查看结果
print("选择的agents:", result["agents"])
print("Prompts:", result["prompts"])
print("推理过程:", result["reasoning"])
```

## 常见问题

### Q: 拉取模型后会自动注册版本吗？

A: 是的！使用 `scripts/download_and_register_model.py` 脚本拉取模型后，会自动注册为初版（例如：`v0_base`）。您可以直接使用这个版本进行推理，无需训练。

### Q: 可以对未训练的模型进行推理吗？

A: 可以！拉取模型并注册为初版后，可以直接使用该版本进行推理：

```bash
python inference/run_inference.py \
    --version v0_base \
    --report "数据集报告..." \
    --available-agents agent1 agent2
```

### Q: 如何知道训练后注册的版本号？

A: 训练完成后会在日志中输出版本号，例如：
```
✓ 模型已注册为版本: v20241212_001
```

也可以运行 `python version/version_manager.py latest` 获取最新版本。

### Q: 可以手动指定版本号吗？

A: 可以，在训练脚本中修改 `version_manager.py` 的调用，或手动注册：

```bash
# 注册训练后的模型
python version/version_manager.py register \
    --model-path ./test_output \
    --version v1.0.0 \
    --description "正式版本 1.0"

# 注册基础模型
python version/version_manager.py register-base \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --version v0_base_custom \
    --description "自定义基础模型版本"
```

### Q: 版本会占用多少磁盘空间？

A: 每个版本会复制完整的模型文件。Qwen2-VL-7B 约 14GB，建议定期清理旧版本。

### Q: 如何备份版本？

A: 版本信息保存在 `model_registry.json`，模型文件在 `model_versions/`。备份这两个即可。

### Q: 推理时如何选择设备？

A: 默认自动选择（优先GPU）。可以手动指定：

```python
inference = ModelInference(version="v20241212_001", device="cpu")
```

## 最佳实践

1. **版本命名**: 使用有意义的描述，便于后续查找
2. **定期清理**: 删除不再需要的旧版本，节省磁盘空间
3. **版本记录**: 在描述中记录训练的关键信息（数据集、参数等）
4. **备份重要版本**: 对重要的模型版本进行额外备份
5. **使用最新版本**: 推理时如果不指定版本，默认使用最新版本


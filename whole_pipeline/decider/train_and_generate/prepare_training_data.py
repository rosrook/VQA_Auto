"""
数据准备脚本：将原始数据转换为 GRPO 训练所需的数据集格式

输入格式：
[
    {
        "report": "数据集报告文本",
        "agent_list": ["agent1", "agent2", ...],  # 所有 report 共享同一个 agent list
        "filtered_datasets": [
            {
                "data_ids": [0, 1, 5, 7, ...],  # 筛选后的数据 ID/索引列表
                "metadata": {...}  # 可选的元数据
            },
            ...
        ]  # 与 report 一一对应
    },
    ...
]

输出格式（符合 GRPO 训练要求）：
[
    {
        "report": "数据集报告文本",
        "ground_truth_agents": ["agent1", "agent2", ...],
        "ground_truth_filtered_data": [0, 1, 5, 7, ...]
    },
    ...
]
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_data(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    加载输入数据文件（支持 JSON 和 JSONL 格式）
    
    参数:
        input_path: 输入文件路径
        
    返回:
        数据字典列表
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    data = []
    
    if input_path.suffix == ".jsonl":
        # JSONL 格式：每行一个 JSON 对象
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过第 {line_num} 行（JSON 解析失败）: {e}")
    else:
        # JSON 格式：整个文件是一个 JSON 数组或对象
        with open(input_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict):
                # 如果是单个字典，转换为列表
                data = [content]
            else:
                raise ValueError(f"不支持的 JSON 格式: 期望 list 或 dict，得到 {type(content)}")
    
    logger.info(f"从 {input_path} 加载了 {len(data)} 条记录")
    return data


def validate_input_format(data: List[Dict[str, Any]]) -> None:
    """
    验证输入数据格式是否正确
    
    参数:
        data: 输入数据列表
        
    抛出:
        ValueError: 如果格式不正确
    """
    required_keys = ["report", "agent_list", "filtered_datasets"]
    
    for i, item in enumerate(data):
        # 检查必需字段
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            raise ValueError(
                f"第 {i+1} 条记录缺少必需字段: {missing_keys}\n"
                f"记录内容: {json.dumps(item, ensure_ascii=False, indent=2)[:200]}..."
            )
        
        # 检查字段类型
        if not isinstance(item["report"], str):
            raise ValueError(f"第 {i+1} 条记录的 'report' 必须是字符串")
        
        if not isinstance(item["agent_list"], list):
            raise ValueError(f"第 {i+1} 条记录的 'agent_list' 必须是列表")
        
        if not isinstance(item["filtered_datasets"], list):
            raise ValueError(f"第 {i+1} 条记录的 'filtered_datasets' 必须是列表")
        
        # 检查 filtered_datasets 中的每个元素
        for j, filtered_data in enumerate(item["filtered_datasets"]):
            if not isinstance(filtered_data, dict):
                raise ValueError(
                    f"第 {i+1} 条记录的第 {j+1} 个 filtered_dataset 必须是字典"
                )
            
            if "data_ids" not in filtered_data:
                raise ValueError(
                    f"第 {i+1} 条记录的第 {j+1} 个 filtered_dataset 缺少 'data_ids' 字段"
                )
            
            if not isinstance(filtered_data["data_ids"], list):
                raise ValueError(
                    f"第 {i+1} 条记录的第 {j+1} 个 filtered_dataset 的 'data_ids' 必须是列表"
                )


def convert_to_training_format(
    input_data: List[Dict[str, Any]],
    agent_list: List[str] = None
) -> List[Dict[str, Any]]:
    """
    将输入数据转换为训练格式
    
    参数:
        input_data: 输入数据列表
        agent_list: 可选的全局 agent 列表（如果输入数据中没有 agent_list 字段）
        
    返回:
        转换后的训练数据列表
    """
    training_data = []
    
    # 如果提供了全局 agent_list，使用它；否则从第一条记录获取
    global_agent_list = agent_list
    if global_agent_list is None and len(input_data) > 0:
        if "agent_list" in input_data[0]:
            global_agent_list = input_data[0]["agent_list"]
            logger.info(f"从输入数据获取 agent_list: {global_agent_list}")
        else:
            raise ValueError("输入数据中没有 'agent_list' 字段，且未提供全局 agent_list")
    
    if not global_agent_list:
        raise ValueError("agent_list 不能为空")
    
    # 转换每条记录
    for i, item in enumerate(input_data):
        report = item["report"]
        
        # 使用全局 agent_list 或当前记录的 agent_list
        current_agent_list = item.get("agent_list", global_agent_list)
        
        # 检查 agent_list 是否一致
        if current_agent_list != global_agent_list:
            logger.warning(
                f"第 {i+1} 条记录的 agent_list 与全局不一致，使用当前记录的 agent_list"
            )
        
        filtered_datasets = item["filtered_datasets"]
        
        # 为每个 filtered_dataset 创建一条训练样本
        for j, filtered_data in enumerate(filtered_datasets):
            data_ids = filtered_data["data_ids"]
            
            training_sample = {
                "report": report,
                "ground_truth_agents": current_agent_list.copy(),  # 复制列表避免引用问题
                "ground_truth_filtered_data": data_ids.copy(),  # 复制列表避免引用问题
            }
            
            # 保留可选的元数据
            if "metadata" in filtered_data:
                training_sample["metadata"] = filtered_data["metadata"]
            
            training_data.append(training_sample)
    
    logger.info(f"转换完成: {len(input_data)} 条输入记录 -> {len(training_data)} 条训练样本")
    return training_data


def save_training_dataset(
    training_data: List[Dict[str, Any]],
    output_path: Union[str, Path],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    save_format: str = "huggingface"
) -> None:
    """
    保存训练数据集
    
    参数:
        training_data: 训练数据列表
        output_path: 输出路径（目录或文件）
        train_split: 训练集比例
        val_split: 验证集比例
        test_split: 测试集比例
        save_format: 保存格式 ("huggingface" 或 "json"/"jsonl")
    """
    output_path = Path(output_path)
    
    # 验证比例
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"数据集划分比例之和必须为 1.0，当前为 {total_split}")
    
    # 划分数据集
    total_samples = len(training_data)
    train_end = int(total_samples * train_split)
    val_end = train_end + int(total_samples * val_split)
    
    train_data = training_data[:train_end]
    val_data = training_data[train_end:val_end] if val_split > 0 else []
    test_data = training_data[val_end:] if test_split > 0 else []
    
    logger.info(
        f"数据集划分: 训练集={len(train_data)}, "
        f"验证集={len(val_data)}, 测试集={len(test_data)}"
    )
    
    if save_format == "huggingface":
        # 保存为 HuggingFace Dataset 格式
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_dict = {}
        if train_data:
            dataset_dict["train"] = Dataset.from_list(train_data)
        if val_data:
            dataset_dict["validation"] = Dataset.from_list(val_data)
        if test_data:
            dataset_dict["test"] = Dataset.from_list(test_data)
        
        dataset = DatasetDict(dataset_dict)
        dataset.save_to_disk(str(output_path))
        
        logger.info(f"已保存 HuggingFace Dataset 到: {output_path}")
        
        # 同时保存 JSON 格式的元数据
        metadata = {
            "total_samples": total_samples,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
        }
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    else:
        # 保存为 JSON 或 JSONL 格式
        if save_format == "jsonl":
            # JSONL 格式：每个 split 一个文件
            output_path.mkdir(parents=True, exist_ok=True)
            
            if train_data:
                with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
                    for item in train_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if val_data:
                with open(output_path / "validation.jsonl", "w", encoding="utf-8") as f:
                    for item in val_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if test_data:
                with open(output_path / "test.jsonl", "w", encoding="utf-8") as f:
                    for item in test_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"已保存 JSONL 格式数据集到: {output_path}")
        
        else:  # json
            # JSON 格式：每个 split 一个文件
            output_path.mkdir(parents=True, exist_ok=True)
            
            if train_data:
                with open(output_path / "train.json", "w", encoding="utf-8") as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            if val_data:
                with open(output_path / "validation.json", "w", encoding="utf-8") as f:
                    json.dump(val_data, f, ensure_ascii=False, indent=2)
            
            if test_data:
                with open(output_path / "test.json", "w", encoding="utf-8") as f:
                    json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存 JSON 格式数据集到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将原始数据转换为 GRPO 训练所需的数据集格式"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="输入文件路径（JSON 或 JSONL 格式）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_data",
        help="输出目录（默认: ./training_data）"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="训练集比例（默认: 0.8）"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="验证集比例（默认: 0.1）"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="测试集比例（默认: 0.1）"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["huggingface", "json", "jsonl"],
        default="huggingface",
        help="保存格式（默认: huggingface）"
    )
    parser.add_argument(
        "--agent-list",
        type=str,
        nargs="+",
        default=None,
        help="全局 agent 列表（如果输入数据中没有 agent_list 字段）"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证输入格式，不进行转换"
    )
    
    args = parser.parse_args()
    
    # 加载输入数据
    logger.info(f"加载输入数据: {args.input_file}")
    input_data = load_input_data(args.input_file)
    
    # 验证格式
    logger.info("验证输入数据格式...")
    try:
        validate_input_format(input_data)
        logger.info("✓ 输入数据格式验证通过")
    except ValueError as e:
        logger.error(f"✗ 输入数据格式验证失败: {e}")
        return 1
    
    if args.validate_only:
        logger.info("仅验证模式，跳过转换")
        return 0
    
    # 转换数据
    logger.info("转换数据格式...")
    training_data = convert_to_training_format(
        input_data,
        agent_list=args.agent_list
    )
    
    # 保存数据集
    logger.info(f"保存训练数据集到: {args.output_dir}")
    save_training_dataset(
        training_data,
        args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        save_format=args.save_format
    )
    
    logger.info("✓ 数据准备完成！")
    return 0


if __name__ == "__main__":
    exit(main())


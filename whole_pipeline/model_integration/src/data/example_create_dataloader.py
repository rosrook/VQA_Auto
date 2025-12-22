"""
从JSONL文件创建VQA DataLoader的完整示例
包含两种方法：使用DataPipeline（推荐）和手动步骤
"""
import sys
from pathlib import Path

# 添加src目录到路径（以便可以直接运行此脚本）
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data.data_pipeline import DataPipeline
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.dataset import VQADataset, create_dataloader

# 安全导入BlipProcessor（如果不存在则使用AutoProcessor）
try:
    from transformers import BlipProcessor
except ImportError:
    from transformers import AutoProcessor as BlipProcessor
    print("警告: 无法导入 BlipProcessor，使用 AutoProcessor 作为回退")


def method1_using_pipeline(config_path: str):
    """
    方法1：使用DataPipeline（推荐）
    
    Args:
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("方法1：使用DataPipeline")
    print("=" * 60)
    
    # 创建并设置数据管线
    print("\n1. 初始化数据管线...")
    pipeline = DataPipeline(config_path)
    pipeline.setup()
    
    # 获取训练DataLoader
    print("\n2. 获取训练DataLoader...")
    train_loader = pipeline.get_train_dataloader()
    
    print(f"\n✓ DataLoader创建成功！")
    print(f"  数据集大小: {len(pipeline.get_dataset('train'))}")
    
    # 测试一个batch
    print("\n3. 测试第一个batch...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Keys: {list(batch.keys())}")
        print(f"  Image shape: {batch['pixel_values'].shape}")
        print(f"  Question input_ids shape: {batch['input_ids'].shape}")
        print(f"  Question attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Answer labels shape: {batch['labels'].shape}")
        
        if batch_idx >= 0:  # 只测试第一个batch
            break
    
    return train_loader


def method2_manual_steps(
    jsonl_path: str,
    image_root: str = None,
    model_name: str = "Salesforce/blip-vqa-base",
    batch_size: int = 16,
    max_length: int = 128
):
    """
    方法2：手动步骤（更灵活）
    
    Args:
        jsonl_path: JSONL文件路径
        image_root: 图像根目录（如果image字段是相对路径）
        model_name: HuggingFace模型名称
        batch_size: 批次大小
        max_length: 最大文本长度
        
    Returns:
        DataLoader对象
    """
    print("\n" + "=" * 60)
    print("方法2：手动步骤")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = DataLoader(jsonl_path)
    raw_data = loader.load()
    print(f"   ✓ 加载了 {len(raw_data)} 条数据")
    if raw_data:
        print(f"   第一条数据字段: {list(raw_data[0].keys())}")
    
    # 2. 预处理数据
    print("\n2. 预处理数据...")
    processor = DataProcessor()
    processed_data = processor.process(raw_data, task_type='vqa')
    print(f"   ✓ 预处理完成，共 {len(processed_data)} 条数据")
    
    # 3. 加载processor
    print(f"\n3. 加载processor: {model_name}...")
    blip_processor = BlipProcessor.from_pretrained(model_name)
    tokenizer = blip_processor.tokenizer
    image_processor = blip_processor.image_processor
    print("   ✓ Processor加载完成")
    
    # 4. 创建Dataset
    print("\n4. 创建Dataset...")
    vqa_dataset = VQADataset(
        data=processed_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length,
        image_field="image",
        question_field="question",
        answer_field="answer",
        image_root=image_root,
        return_raw_image=False
    )
    print(f"   ✓ Dataset创建成功，样本数: {len(vqa_dataset)}")
    
    # 5. 创建DataLoader
    print("\n5. 创建DataLoader...")
    dataloader = create_dataloader(
        dataset=vqa_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设为0避免多进程问题（调试时）
        pin_memory=True
    )
    print("   ✓ DataLoader创建成功！")
    
    # 6. 测试
    print("\n6. 测试DataLoader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Image shape: {batch['pixel_values'].shape}")
        print(f"  Question input_ids shape: {batch['input_ids'].shape}")
        print(f"  Question attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Answer labels shape: {batch['labels'].shape}")
        
        if batch_idx >= 0:
            break
    
    return dataloader


def create_simple_dataloader(jsonl_path: str, **kwargs):
    """
    快速创建DataLoader的便捷函数
    
    Args:
        jsonl_path: JSONL文件路径
        **kwargs: 其他参数（image_root, model_name, batch_size等）
        
    Returns:
        DataLoader对象
    """
    # 默认参数
    defaults = {
        'image_root': None,
        'model_name': "Salesforce/blip-vqa-base",
        'batch_size': 16,
        'max_length': 128
    }
    defaults.update(kwargs)
    
    return method2_manual_steps(jsonl_path, **defaults)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从JSONL文件创建VQA DataLoader")
    parser.add_argument("--method", type=str, choices=["1", "2", "both"], default="both",
                       help="使用方法：1=DataPipeline, 2=手动步骤, both=两种都演示")
    parser.add_argument("--jsonl", type=str, required=True,
                       help="JSONL文件路径")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（方法1需要）")
    parser.add_argument("--image_root", type=str, default=None,
                       help="图像根目录（如果image字段是相对路径）")
    parser.add_argument("--model", type=str, default="Salesforce/blip-vqa-base",
                       help="HuggingFace模型名称")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    
    args = parser.parse_args()
    
    if args.method in ["1", "both"]:
        if args.config is None:
            print("⚠️  使用方法1需要配置文件，请使用 --config 指定")
        else:
            method1_using_pipeline(args.config)
    
    if args.method in ["2", "both"]:
        method2_manual_steps(
            jsonl_path=args.jsonl,
            image_root=args.image_root,
            model_name=args.model,
            batch_size=args.batch_size
        )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


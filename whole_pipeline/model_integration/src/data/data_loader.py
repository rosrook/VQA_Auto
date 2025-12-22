"""
数据加载模块
支持多种数据格式的加载和基本验证
专门优化VQA和Image Captioning任务
"""
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class DataLoader:
    """通用数据加载器，支持视觉-语言任务"""
    
    SUPPORTED_FORMATS = ['.json', '.jsonl', '.csv', '.txt', '.tsv', '.parquet']
    
    def __init__(self, data_path: Union[str, Path]):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        self.file_extension = self.data_path.suffix.lower()
        if self.file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {self.file_extension}. "
                f"支持的格式: {self.SUPPORTED_FORMATS}"
            )
    
    def load(self) -> List[Dict]:
        """
        根据文件格式加载数据
        
        Returns:
            数据列表，每个元素是一个字典
        """
        logger.info(f"加载数据文件: {self.data_path}")
        
        if self.file_extension == '.json':
            data = self._load_json()
        elif self.file_extension == '.jsonl':
            data = self._load_jsonl()
        elif self.file_extension in ['.csv', '.tsv']:
            data = self._load_csv()
        elif self.file_extension == '.txt':
            data = self._load_txt()
        elif self.file_extension == '.parquet':
            data = self._load_parquet()
        else:
            raise ValueError(f"未实现的文件格式: {self.file_extension}")
        
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    
    def _load_json(self) -> List[Dict]:
        """加载JSON格式数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个字典，转换为列表
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def _load_jsonl(self) -> List[Dict]:
        """加载JSONL格式数据（每行一个JSON对象）"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
        return data
    
    def _load_csv(self) -> List[Dict]:
        """加载CSV/TSV格式数据"""
        delimiter = '\t' if self.file_extension == '.tsv' else ','
        data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(dict(row))
        
        return data
    
    def _load_txt(self) -> List[Dict]:
        """
        加载TXT格式数据
        假设每行是一个文本样本，自动添加索引
        """
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    data.append({'text': line, 'id': idx})
        return data
    
    def _load_parquet(self) -> List[Dict]:
        """加载Parquet格式数据"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装pandas来读取parquet文件: pip install pandas pyarrow")
        
        df = pd.read_parquet(self.data_path)
        # 将DataFrame转换为字典列表
        data = df.to_dict('records')
        
        # 处理NaN值
        for item in data:
            for key, value in item.items():
                if pd.isna(value):
                    item[key] = None
        
        return data
    
    @staticmethod
    def validate_data(
        data: List[Dict],
        required_fields: Optional[List[str]] = None
    ) -> bool:
        """
        验证数据格式
        
        Args:
            data: 数据列表
            required_fields: 必需的字段列表
            
        Returns:
            验证是否通过
        """
        if not data:
            logger.warning("数据为空")
            return False
        
        if required_fields:
            for idx, item in enumerate(data):
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    logger.error(
                        f"第 {idx} 条数据缺少字段: {missing_fields}. "
                        f"数据内容: {item}"
                    )
                    return False
        
        logger.info("数据验证通过")
        return True
    
    @staticmethod
    def get_data_statistics(data: List[Dict]) -> Dict:
        """
        获取数据统计信息
        
        Args:
            data: 数据列表
            
        Returns:
            统计信息字典
        """
        if not data:
            return {'total': 0}
        
        stats = {
            'total': len(data),
            'fields': list(data[0].keys()) if data else [],
            'sample': data[0] if data else None
        }
        
        # 统计文本长度（如果有text字段）
        if 'text' in data[0]:
            text_lengths = [len(item['text']) for item in data if 'text' in item]
            stats['text_length'] = {
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0,
                'avg': sum(text_lengths) / len(text_lengths) if text_lengths else 0
            }
        
        # 统计标签分布（如果有label字段）
        if 'label' in data[0]:
            from collections import Counter
            label_counts = Counter(item['label'] for item in data if 'label' in item)
            stats['label_distribution'] = dict(label_counts)
        
        # 统计图像信息（如果有image相关字段）
        image_fields = ['image', 'image_path', 'image_base64', 'image_url']
        for field in image_fields:
            if field in data[0]:
                stats['has_images'] = True
                stats['image_field'] = field
                break
        
        # 统计VQA问题长度
        if 'question' in data[0]:
            question_lengths = [len(item['question']) for item in data if 'question' in item]
            stats['question_length'] = {
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0,
                'avg': sum(question_lengths) / len(question_lengths) if question_lengths else 0
            }
        
        # 统计Caption长度
        if 'caption' in data[0]:
            caption_lengths = [len(item['caption']) for item in data if 'caption' in item]
            stats['caption_length'] = {
                'min': min(caption_lengths) if caption_lengths else 0,
                'max': max(caption_lengths) if caption_lengths else 0,
                'avg': sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0
            }
        
        return stats


class MultiFileDataLoader:
    """多文件数据加载器"""
    
    def __init__(self, data_dir: Union[str, Path], pattern: str = '*'):
        """
        初始化多文件数据加载器
        
        Args:
            data_dir: 数据目录
            pattern: 文件匹配模式
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        self.pattern = pattern
    
    def load_all(self) -> List[Dict]:
        """
        加载目录下所有匹配的文件
        
        Returns:
            合并后的数据列表
        """
        all_data = []
        file_list = list(self.data_dir.glob(self.pattern))
        
        if not file_list:
            logger.warning(f"未找到匹配的文件: {self.data_dir}/{self.pattern}")
            return all_data
        
        logger.info(f"找到 {len(file_list)} 个文件")
        
        for file_path in file_list:
            try:
                loader = DataLoader(file_path)
                data = loader.load()
                all_data.extend(data)
                logger.info(f"从 {file_path.name} 加载 {len(data)} 条数据")
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
        
        logger.info(f"总共加载 {len(all_data)} 条数据")
        return all_data


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: 加载VQA数据
    # vqa_loader = DataLoader("data/vqa_train.parquet")
    # vqa_data = vqa_loader.load()
    # DataLoader.validate_data(vqa_data, required_fields=['image_path', 'question', 'answer'])
    
    # 示例2: 加载Image Caption数据
    # caption_loader = DataLoader("data/captions.jsonl")
    # caption_data = caption_loader.load()
    # DataLoader.validate_data(caption_data, required_fields=['image_path', 'caption'])
    
    # 示例3: 加载多个文件
    # multi_loader = MultiFileDataLoader("data/raw", pattern="*.parquet")
    # all_data = multi_loader.load_all()
    
    # 示例4: 获取统计信息
    # stats = DataLoader.get_data_statistics(vqa_data)
    # print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("DataLoader模块加载完成 - 支持VQA和Image Captioning任务")
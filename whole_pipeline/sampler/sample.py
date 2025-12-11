"""
聚类结果采样器

从聚类结果JSON文件中，对每一类进行采样，并转换为evaluator需要的DataSample格式。
"""

import json
import base64
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """采样策略"""
    RANDOM = "random"  # 随机采样
    # 可以扩展其他策略：
    # FIRST_N = "first_n"  # 前N个
    # LAST_N = "last_n"  # 后N个
    # STRATIFIED = "stratified"  # 分层采样
    # DIVERSITY = "diversity"  # 多样性采样


@dataclass
class DataSample:
    """数据样本（与evaluator.py中的格式一致）"""
    sample_id: str
    image_base64: Optional[List[str]] = None
    dialogue: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict] = None


class ClusterSampler:
    """聚类结果采样器"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化采样器
        
        参数:
            seed: 随机种子，用于可重复采样
        """
        if seed is not None:
            random.seed(seed)
            logger.info(f"设置随机种子: {seed}")
    
    def _convert_buffer_to_base64(self, buffer: Any) -> str:
        """
        将image buffer转换为base64字符串
        
        参数:
            buffer: 可能是bytes、字符串形式的bytes表示等
            
        返回:
            base64编码的字符串
        """
        try:
            # 如果buffer是字符串形式的bytes表示，如 "b'...'"
            if isinstance(buffer, str):
                # 尝试解析字符串形式的bytes
                if buffer.startswith("b'") or buffer.startswith('b"'):
                    # 移除 b' 和 '
                    buffer_str = buffer[2:-1]
                    # 处理转义字符
                    buffer_bytes = buffer_str.encode('utf-8').decode('unicode_escape').encode('latin1')
                else:
                    # 假设已经是base64字符串
                    return buffer
            elif isinstance(buffer, bytes):
                buffer_bytes = buffer
            else:
                logger.warning(f"未知的buffer类型: {type(buffer)}, 尝试转换为bytes")
                buffer_bytes = bytes(buffer)
            
            # 转换为base64
            base64_str = base64.b64encode(buffer_bytes).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"转换buffer到base64失败: {e}")
            return ""
    
    def _convert_sample_to_datasample(self, sample: Dict) -> Optional[DataSample]:
        """
        将聚类结果中的sample转换为DataSample格式
        
        参数:
            sample: 聚类结果中的单个样本
            
        返回:
            DataSample对象，如果转换失败返回None
        """
        try:
            sample_id = str(sample.get('sample_id', ''))
            data = sample.get('data', {})
            
            # 转换图片
            image_base64_list = []
            image_buffer_list = data.get('image_buffer_list', [])
            for img_item in image_buffer_list:
                buffer = img_item.get('buffer')
                if buffer:
                    base64_str = self._convert_buffer_to_base64(buffer)
                    if base64_str:
                        image_base64_list.append(base64_str)
            
            # 转换对话
            dialogue_list = []
            dialogue_data = data.get('dialogue', [])
            for turn in dialogue_data:
                question = turn.get('question', '')
                answer = turn.get('answer', '')
                if question and answer:
                    dialogue_list.append({
                        'question': question,
                        'answer': answer
                    })
            
            # 构建metadata
            metadata = {
                'cluster_id': sample.get('cluster_id'),
                'source': data.get('source'),
                'task_type': data.get('task_type'),
                'image_id_list': data.get('image_id_list', [])
            }
            
            return DataSample(
                sample_id=sample_id,
                image_base64=image_base64_list if image_base64_list else None,
                dialogue=dialogue_list if dialogue_list else None,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"转换sample失败 (sample_id={sample.get('sample_id', 'unknown')}): {e}")
            return None
    
    def sample_cluster(
        self,
        cluster: Dict,
        sample_size: int,
        strategy: SamplingStrategy = SamplingStrategy.RANDOM
    ) -> List[DataSample]:
        """
        从单个cluster中采样
        
        参数:
            cluster: 聚类结果中的单个cluster
            sample_size: 采样数量
            strategy: 采样策略
            
        返回:
            DataSample列表
        """
        cluster_id = cluster.get('cluster_id', 'unknown')
        samples = cluster.get('samples', [])
        total_size = len(samples)
        
        logger.info(f"Cluster {cluster_id}: 总样本数={total_size}, 需要采样={sample_size}")
        
        if total_size == 0:
            logger.warning(f"Cluster {cluster_id} 没有样本")
            return []
        
        # 如果需要的采样数大于等于总数，返回全部
        if sample_size >= total_size:
            logger.info(f"Cluster {cluster_id}: 采样数 >= 总数，返回全部样本")
            selected_samples = samples
        else:
            # 根据策略选择样本
            if strategy == SamplingStrategy.RANDOM:
                selected_samples = random.sample(samples, sample_size)
            else:
                # 默认使用随机采样
                selected_samples = random.sample(samples, sample_size)
        
        # 转换为DataSample格式
        data_samples = []
        for sample in selected_samples:
            data_sample = self._convert_sample_to_datasample(sample)
            if data_sample:
                data_samples.append(data_sample)
            else:
                logger.warning(f"Cluster {cluster_id}: 跳过无效样本 (sample_id={sample.get('sample_id', 'unknown')})")
        
        logger.info(f"Cluster {cluster_id}: 成功采样 {len(data_samples)} 个样本")
        return data_samples
    
    def sample_from_clustered_results(
        self,
        clustered_json_path: str,
        sample_size_per_cluster: int,
        output_dir: str,
        strategy: SamplingStrategy = SamplingStrategy.RANDOM,
        output_prefix: str = "cluster"
    ) -> Dict[int, str]:
        """
        从聚类结果文件中采样，每个cluster单独输出文件
        
        参数:
            clustered_json_path: 聚类结果JSON文件路径
            sample_size_per_cluster: 每个cluster的采样数量
            output_dir: 输出目录
            strategy: 采样策略
            output_prefix: 输出文件前缀
            
        返回:
            Dict[cluster_id, output_file_path]
        """
        # 读取聚类结果
        logger.info(f"读取聚类结果文件: {clustered_json_path}")
        with open(clustered_json_path, 'r', encoding='utf-8') as f:
            clustered_data = json.load(f)
        
        clusters = clustered_data.get('clusters', [])
        metadata = clustered_data.get('metadata', {})
        
        logger.info(f"总cluster数: {len(clusters)}")
        logger.info(f"元数据: {metadata}")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 对每个cluster进行采样
        output_files = {}
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            
            # 采样
            data_samples = self.sample_cluster(cluster, sample_size_per_cluster, strategy)
            
            if not data_samples:
                logger.warning(f"Cluster {cluster_id}: 没有成功采样的样本，跳过")
                continue
            
            # 转换为可序列化格式
            serializable_samples = [asdict(ds) for ds in data_samples]
            
            # 保存为JSON文件
            output_file = output_path / f"{output_prefix}_{cluster_id}_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'cluster_id': cluster_id,
                    'cluster_size': cluster.get('size', 0),
                    'sample_size': len(data_samples),
                    'sampling_strategy': strategy.value,
                    'samples': serializable_samples
                }, f, ensure_ascii=False, indent=2)
            
            output_files[cluster_id] = str(output_file)
            logger.info(f"Cluster {cluster_id}: 采样结果已保存到 {output_file}")
        
        return output_files


def main():
    """主函数 - 示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从聚类结果中采样')
    parser.add_argument('--input', type=str, required=True,
                       help='聚类结果JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='sampled_results',
                       help='输出目录')
    parser.add_argument('--sample_size', type=int, default=3,
                       help='每个cluster的采样数量')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random'],
                       help='采样策略')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--prefix', type=str, default='cluster',
                       help='输出文件前缀')
    
    args = parser.parse_args()
    
    # 创建采样器
    sampler = ClusterSampler(seed=args.seed)
    
    # 选择采样策略
    strategy = SamplingStrategy(args.strategy)
    
    # 执行采样
    output_files = sampler.sample_from_clustered_results(
        clustered_json_path=args.input,
        sample_size_per_cluster=args.sample_size,
        output_dir=args.output_dir,
        strategy=strategy,
        output_prefix=args.prefix
    )
    
    # 输出结果摘要
    print("\n" + "=" * 80)
    print("采样完成!")
    print("=" * 80)
    print(f"共处理 {len(output_files)} 个cluster")
    print("\n输出文件:")
    for cluster_id, file_path in sorted(output_files.items()):
        print(f"  Cluster {cluster_id}: {file_path}")


if __name__ == "__main__":
    main()


# python sample.py \
#   --input /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/b_clustered_data/vqa_clustered_results_final.json \
#   --output_dir /home/zhuxuzhou/VQA_Auto/whole_pipeline/data/c_sampled_data \
#   --sample_size 3 \
#   --strategy random \
#   --seed 42 \
#   --prefix cluster

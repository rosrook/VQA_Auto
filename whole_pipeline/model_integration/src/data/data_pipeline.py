"""
完整的数据加载和处理管线
整合所有数据处理模块，专门优化VQA和Image Captioning任务
支持大规模数据集的内存优化
"""
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor

# 使用绝对导入（相对于src目录）
from data.data_loader import DataLoader, MultiFileDataLoader
from data.data_processor import DataProcessor
from data.dataset import (
    VQADataset,
    ImageCaptioningDataset,
    ClassificationDataset,
    Seq2SeqDataset,
    CausalLMDataset,
    create_dataloader
)
from data.dataset_optimized import (
    LazyLoadVQADataset,
    StreamingVQADataset,
    MemoryMappedVQADataset,
    create_optimized_dataloader
)
from data.memory_utils import (
    MemoryMonitor,
    DatasetOptimizer,
    DatasetSizeEstimator
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """完整的数据处理管线"""
    
    def __init__(self, config_path: str):
        """
        初始化数据管线
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.image_processor = None
        self.datasets = {}
        self.dataloaders = {}
        
        # 设置日志
        logging.basicConfig(
            level=self.config.get('logging', {}).get('level', 'INFO')
        )
        
        logger.info("数据管线初始化完成")
        logger.info(f"任务类型: {self.config.get('task_type', 'unknown')}")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup(self):
        """设置管线：加载tokenizer、image processor和数据"""
        logger.info("开始设置数据管线...")
        
        # 0. 内存检查和优化建议
        self._check_memory_and_optimize()
        
        # 1. 加载tokenizer
        self._load_tokenizer()
        
        # 2. 加载image processor（如果是视觉任务）
        task_type = self.config.get('task_type', '')
        if task_type in ['vqa', 'image_captioning']:
            self._load_image_processor()
        
        # 3. 加载和处理数据
        self._load_and_process_data()
        
        # 4. 创建PyTorch数据集
        self._create_datasets()
        
        # 5. 创建DataLoader
        self._create_dataloaders()
        
        # 6. 打印最终内存使用情况
        if self.config.get('logging', {}).get('log_statistics', True):
            from data.memory_utils import MemoryMonitor
            MemoryMonitor.print_memory_info()
        
        logger.info("数据管线设置完成")
    
    def _load_tokenizer(self):
        """加载tokenizer，支持多种类型（BLIP、CLIP等）"""
        tokenizer_config = self.config.get('tokenizer', {})
        tokenizer_name = tokenizer_config.get('name', 'bert-base-uncased')
        tokenizer_type = tokenizer_config.get('type', None)  # 'blip', 'clip', 'auto'等
        
        logger.info(f"加载tokenizer: {tokenizer_name} (类型: {tokenizer_type or 'auto'})")
        
        # 如果指定了类型，使用DataProcessor来加载
        if tokenizer_type and tokenizer_type != 'auto':
            try:
                from data.data_processor import DataProcessor
                processor_info = DataProcessor.PROCESSOR_REGISTRY.get(tokenizer_type)
                if processor_info:
                    tokenizer_class = processor_info['tokenizer_class']
                    self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
                    logger.info(f"使用 {tokenizer_type} tokenizer: {type(self.tokenizer).__name__}")
                else:
                    raise ValueError(f"未知的tokenizer类型: {tokenizer_type}")
            except Exception as e:
                logger.warning(f"使用指定类型加载tokenizer失败: {e}，回退到AutoTokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # 自动检测或使用AutoTokenizer
            try:
                # 尝试使用DataProcessor自动检测
                from data.data_processor import DataProcessor
                temp_processor = DataProcessor({'processor_name': tokenizer_name})
                if temp_processor.tokenizer:
                    self.tokenizer = temp_processor.tokenizer
                    logger.info(f"自动检测到tokenizer: {type(self.tokenizer).__name__}")
                else:
                    # 回退到AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    logger.info("使用AutoTokenizer")
            except Exception as e:
                logger.warning(f"自动检测失败: {e}，使用AutoTokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 确保有pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                logger.warning("无法设置pad_token，某些模型可能需要手动设置")
    
    def _load_image_processor(self):
        """加载图像处理器，支持多种类型（BLIP、CLIP等）"""
        image_processor_config = self.config.get('image_processor', {})
        processor_name = image_processor_config.get('name', 'google/vit-base-patch16-224')
        processor_type = image_processor_config.get('type', None)  # 'blip', 'clip', 'auto'等
        
        logger.info(f"加载image processor: {processor_name} (类型: {processor_type or 'auto'})")
        
        # 如果指定了类型，使用DataProcessor来加载
        if processor_type and processor_type != 'auto':
            try:
                from data.data_processor import DataProcessor
                processor_info = DataProcessor.PROCESSOR_REGISTRY.get(processor_type)
                if processor_info:
                    # 尝试加载完整的processor（包含image_processor和tokenizer）
                    processor_class = processor_info['processor_class']
                    try:
                        full_processor = processor_class.from_pretrained(processor_name)
                        self.image_processor = full_processor.image_processor if hasattr(full_processor, 'image_processor') else full_processor
                        logger.info(f"使用 {processor_type} processor: {type(self.image_processor).__name__}")
                    except:
                        # 如果完整processor失败，尝试只加载image_processor
                        image_processor_class = processor_info['image_processor_class']
                        self.image_processor = image_processor_class.from_pretrained(processor_name)
                        logger.info(f"使用 {processor_type} image processor: {type(self.image_processor).__name__}")
                else:
                    raise ValueError(f"未知的processor类型: {processor_type}")
            except Exception as e:
                logger.warning(f"使用指定类型加载processor失败: {e}，回退到AutoProcessor")
                self._load_image_processor_fallback(processor_name)
        else:
            # 自动检测或使用AutoProcessor
            try:
                # 尝试使用DataProcessor自动检测
                from data.data_processor import DataProcessor
                temp_processor = DataProcessor({'processor_name': processor_name})
                if temp_processor.image_processor:
                    self.image_processor = temp_processor.image_processor
                    logger.info(f"自动检测到image processor: {type(self.image_processor).__name__}")
                elif temp_processor.processor and hasattr(temp_processor.processor, 'image_processor'):
                    self.image_processor = temp_processor.processor.image_processor
                    logger.info(f"自动检测到processor: {type(self.image_processor).__name__}")
                else:
                    # 回退到AutoProcessor
                    self._load_image_processor_fallback(processor_name)
            except Exception as e:
                logger.warning(f"自动检测失败: {e}，使用AutoProcessor")
                self._load_image_processor_fallback(processor_name)
    
    def _load_image_processor_fallback(self, processor_name: str):
        """回退方法：使用AutoProcessor或AutoImageProcessor"""
        try:
            # 尝试加载完整的processor（如CLIP）
            self.image_processor = AutoProcessor.from_pretrained(processor_name)
            logger.info("使用AutoProcessor")
        except:
            # 如果失败，尝试只加载图像processor
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(processor_name)
                logger.info("使用AutoImageProcessor")
            except Exception as e:
                logger.warning(f"无法加载image processor: {e}，将使用默认处理")
                self.image_processor = None
    
    def _check_memory_and_optimize(self):
        """检查内存并给出优化建议"""
        from data.memory_utils import MemoryMonitor, DatasetOptimizer
        
        # 打印当前内存
        logger.info("检查系统资源...")
        MemoryMonitor.print_memory_info()
        
        # 如果配置了数据路径，分析并给出建议
        data_paths = self.config.get('data_paths', {})
        if 'train' in data_paths:
            try:
                recommendations = DatasetOptimizer.analyze_and_recommend(
                    data_path=data_paths['train'],
                    batch_size=self.config.get('dataloader', {}).get('batch_size', 16)
                )
                DatasetOptimizer.print_recommendations(recommendations)
                
                # 保存建议供后续使用
                self._optimization_strategy = recommendations['recommended_strategy']
            except Exception as e:
                logger.warning(f"无法分析数据集: {e}")
                self._optimization_strategy = 'medium_dataset'
        else:
            self._optimization_strategy = 'medium_dataset'
    
    def _load_and_process_data(self):
        """加载和预处理数据"""
        from data.data_loader import DataLoader
        from data.data_processor import DataProcessor
        
        data_paths = self.config.get('data_paths', {})
        preprocessing_config = self.config.get('preprocessing', {})
        
        # 初始化处理器
        processor = DataProcessor(preprocessing_config)
        
        # 加载训练数据
        if 'train' in data_paths:
            logger.info("加载训练数据...")
            loader = DataLoader(data_paths['train'])
            train_data = loader.load()
            
            # 验证数据
            required_fields = self._get_required_fields()
            if not DataLoader.validate_data(train_data, required_fields):
                raise ValueError("训练数据验证失败")
            
            # 预处理
            train_data = processor.process(train_data)
            
            # 数据平衡（如果需要）
            balance_config = self.config.get('balance', {})
            if balance_config.get('enabled', False):
                train_data = processor.balance_dataset(
                    train_data,
                    label_field=balance_config.get('label_field', 'label'),
                    strategy=balance_config.get('strategy', 'undersample')
                )
            
            self.raw_data = {'train': train_data}
            
            # 打印数据统计
            if self.config.get('logging', {}).get('log_statistics', True):
                stats = DataLoader.get_data_statistics(train_data)
                logger.info(f"训练数据统计: {stats}")
        
        # 加载验证数据（如果提供了且不为None）
        validation_path = data_paths.get('validation')
        if validation_path is not None:
            logger.info("加载验证数据...")
            loader = DataLoader(validation_path)
            val_data = loader.load()
            val_data = processor.process(val_data)
            self.raw_data['validation'] = val_data
        
        # 加载测试数据（如果提供了且不为None）
        test_path = data_paths.get('test')
        if test_path is not None:
            logger.info("加载测试数据...")
            loader = DataLoader(test_path)
            test_data = loader.load()
            test_data = processor.process(test_data)
            self.raw_data['test'] = test_data
        
        # 检查是否需要自动分割数据集
        split_config = self.config.get('data_split', {})
        auto_split = split_config.get('auto_split', True)  # 默认自动分割
        
        # 判断是否缺少验证/测试集
        has_validation = 'validation' in self.raw_data
        has_test = 'test' in self.raw_data
        
        if not has_validation or not has_test:
            if auto_split:
                # 自动分割数据集
                logger.info("缺少验证/测试集，自动分割数据集...")
                train_data, val_data, test_data = DataProcessor.split_data(
                    self.raw_data['train'],
                    train_ratio=split_config.get('train_ratio', 0.8),
                    val_ratio=split_config.get('val_ratio', 0.1),
                    test_ratio=split_config.get('test_ratio', 0.1),
                    shuffle=split_config.get('shuffle', True),
                    random_seed=split_config.get('random_seed', 42),
                    stratify_by=split_config.get('stratify_by')
                )
                
                self.raw_data = {
                    'train': train_data,
                    'validation': val_data,
                    'test': test_data
                }
                logger.info(f"数据分割完成: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
            else:
                # 不进行分割，使用全部数据作为训练集
                logger.info("auto_split=False，不进行数据分割，使用全部数据作为训练集")
                logger.info(f"训练集大小: {len(self.raw_data['train'])}")
                # self.raw_data 已经包含 'train'，不需要额外操作
        else:
            logger.info("已提供完整的训练/验证/测试集，不进行自动分割")
    
    def _get_required_fields(self):
        """获取必需的数据字段"""
        task_type = self.config.get('task_type', 'classification')
        
        if task_type == 'vqa':
            vqa_config = self.config.get('vqa', {})
            data_fields = vqa_config.get('data_fields', {})
            return [
                data_fields.get('image_field', 'image'),
                data_fields.get('question_field', 'question'),
                data_fields.get('answer_field', 'answer')
            ]
        
        elif task_type == 'image_captioning':
            caption_config = self.config.get('image_captioning', {})
            data_fields = caption_config.get('data_fields', {})
            return [
                data_fields.get('image_field', 'image'),
                data_fields.get('caption_field', 'caption')
            ]
        
        elif task_type == 'classification':
            classification_config = self.config.get('classification', {})
            data_fields = classification_config.get('data_fields', {})
            return [
                data_fields.get('text_field', 'text'),
                data_fields.get('label_field', 'label')
            ]
        
        elif task_type == 'seq2seq':
            seq2seq_config = self.config.get('seq2seq', {})
            data_fields = seq2seq_config.get('data_fields', {})
            return [
                data_fields.get('source_field', 'source'),
                data_fields.get('target_field', 'target')
            ]
        
        elif task_type == 'causal_lm':
            causal_config = self.config.get('causal_lm', {})
            data_fields = causal_config.get('data_fields', {})
            return [data_fields.get('text_field', 'text')]
        
        else:
            return []
    
    def _create_datasets(self):
        """创建PyTorch数据集（根据优化策略选择）"""
        from data.dataset import (
            VQADataset,
            ImageCaptioningDataset,
            ClassificationDataset,
            Seq2SeqDataset,
            CausalLMDataset
        )
        from data.dataset_optimized import LazyLoadVQADataset, StreamingVQADataset
        
        task_type = self.config.get('task_type', 'classification')
        tokenizer_config = self.config.get('tokenizer', {})
        max_length = tokenizer_config.get('max_length', 512)
        
        # 检查是否使用流式数据集
        use_streaming = self.config.get('optimization', {}).get('use_streaming', False)
        
        dataset_class = None
        dataset_kwargs = {
            'tokenizer': self.tokenizer,
        }
        
        # VQA任务
        if task_type == 'vqa':
            vqa_config = self.config.get('vqa', {})
            data_fields = vqa_config.get('data_fields', {})
            image_config = self.config.get('image', {})
            optimization = self.config.get('optimization', {})
            
            # 根据优化策略选择数据集类
            if use_streaming or self._optimization_strategy == 'large_dataset':
                logger.info("使用StreamingVQADataset（流式加载）")
                # 流式数据集需要特殊处理
                self._create_streaming_datasets()
                return
            
            elif self._optimization_strategy == 'small_dataset':
                logger.info("使用标准VQADataset")
                dataset_class = VQADataset
                dataset_kwargs.update({
                    'image_processor': self.image_processor,
                    'max_length': vqa_config.get('max_question_length', max_length),
                    'image_field': data_fields.get('image_field', 'image'),
                    'question_field': data_fields.get('question_field', 'question'),
                    'answer_field': data_fields.get('answer_field', 'answer'),
                    'image_root': image_config.get('root_dir'),
                    'return_raw_image': self.config.get('special', {}).get('return_raw_image', False)
                })
            
            else:  # medium_dataset
                logger.info("使用LazyLoadVQADataset（懒加载+缓存）")
                dataset_class = LazyLoadVQADataset
                dataset_kwargs.update({
                    'image_processor': self.image_processor,
                    'max_length': vqa_config.get('max_question_length', max_length),
                    'image_field': data_fields.get('image_field', 'image'),
                    'question_field': data_fields.get('question_field', 'question'),
                    'answer_field': data_fields.get('answer_field', 'answer'),
                    'image_root': image_config.get('root_dir'),
                    'cache_images': optimization.get('cache_images', True),
                    'preload_images': False
                })
        
        # Image Captioning任务
        elif task_type == 'image_captioning':
            caption_config = self.config.get('image_captioning', {})
            data_fields = caption_config.get('data_fields', {})
            image_config = self.config.get('image', {})
            
            dataset_class = ImageCaptioningDataset
            dataset_kwargs.update({
                'image_processor': self.image_processor,
                'max_length': caption_config.get('max_caption_length', 128),
                'image_field': data_fields.get('image_field', 'image'),
                'caption_field': data_fields.get('caption_field', 'caption'),
                'image_root': image_config.get('root_dir'),
                'multiple_captions': caption_config.get('multiple_captions', False),
                'return_raw_image': self.config.get('special', {}).get('return_raw_image', False)
            })
        
        # 文本分类任务
        elif task_type == 'classification':
            classification_config = self.config.get('classification', {})
            data_fields = classification_config.get('data_fields', {})
            
            dataset_class = ClassificationDataset
            dataset_kwargs.update({
                'max_length': max_length,
                'text_field': data_fields.get('text_field', 'text'),
                'label_field': data_fields.get('label_field', 'label'),
                'num_labels': classification_config.get('num_labels')
            })
        
        # Seq2Seq任务
        elif task_type == 'seq2seq':
            seq2seq_config = self.config.get('seq2seq', {})
            data_fields = seq2seq_config.get('data_fields', {})
            
            dataset_class = Seq2SeqDataset
            dataset_kwargs.update({
                'max_source_length': seq2seq_config.get('max_source_length', max_length),
                'max_target_length': seq2seq_config.get('max_target_length', max_length),
                'source_field': data_fields.get('source_field', 'source'),
                'target_field': data_fields.get('target_field', 'target')
            })
        
        # 因果语言模型任务
        elif task_type == 'causal_lm':
            causal_config = self.config.get('causal_lm', {})
            data_fields = causal_config.get('data_fields', {})
            
            dataset_class = CausalLMDataset
            dataset_kwargs.update({
                'max_length': causal_config.get('max_length', max_length),
                'text_field': data_fields.get('text_field', 'text'),
                'instruction_field': data_fields.get('instruction_field'),
                'response_field': data_fields.get('response_field')
            })
        
        # 创建各个split的数据集
        for split_name, data in self.raw_data.items():
            logger.info(f"创建{split_name}数据集，样本数: {len(data)}...")
            self.datasets[split_name] = dataset_class(
                data=data,
                **dataset_kwargs
            )
    
    def _create_streaming_datasets(self):
        """创建流式数据集"""
        from data.dataset_optimized import StreamingVQADataset
        
        vqa_config = self.config.get('vqa', {})
        data_fields = vqa_config.get('data_fields', {})
        image_config = self.config.get('image', {})
        data_paths = self.config.get('data_paths', {})
        
        dataset_kwargs = {
            'tokenizer': self.tokenizer,
            'image_processor': self.image_processor,
            'max_length': vqa_config.get('max_question_length', 512),
            'image_field': data_fields.get('image_field', 'image'),
            'question_field': data_fields.get('question_field', 'question'),
            'answer_field': data_fields.get('answer_field', 'answer'),
            'image_root': image_config.get('root_dir'),
            'file_format': 'jsonl' if data_paths.get('train', '').endswith('.jsonl') else 'parquet',
            'skip_errors': True
        }
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in data_paths:
                logger.info(f"创建流式{split_name}数据集...")
                self.datasets[split_name] = StreamingVQADataset(
                    data_path=data_paths[split_name],
                    **dataset_kwargs
                )
    
    def _create_dataloaders(self):
        """创建DataLoader"""
        from data.dataset import create_dataloader
        
        dataloader_config = self.config.get('dataloader', {})
        
        # 获取tokenizer和vocab_size（用于safe_collate_fn）
        tokenizer = None
        vocab_size = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenizer = self.tokenizer
            # 尝试获取vocab_size
            if hasattr(tokenizer, 'vocab_size'):
                vocab_size = tokenizer.vocab_size
            elif hasattr(tokenizer, '__len__'):
                try:
                    vocab_size = len(tokenizer)
                except:
                    pass
        
        for split_name, dataset in self.datasets.items():
            # 训练集打乱，验证/测试集不打乱
            shuffle = dataloader_config.get('shuffle', True) if split_name == 'train' else False
            
            logger.info(f"创建{split_name} DataLoader...")
            
            # 如果dataset有tokenizer，优先使用dataset的tokenizer
            dataset_tokenizer = tokenizer
            if hasattr(dataset, 'tokenizer') and dataset.tokenizer is not None:
                dataset_tokenizer = dataset.tokenizer
            
            self.dataloaders[split_name] = create_dataloader(
                dataset=dataset,
                batch_size=dataloader_config.get('batch_size', 8),
                shuffle=shuffle,
                num_workers=dataloader_config.get('num_workers', 0),
                pin_memory=dataloader_config.get('pin_memory', True),
                tokenizer=dataset_tokenizer,
                vocab_size=vocab_size,
                use_safe_collate=True  # 启用安全的collate函数
            )
    
    def get_dataloader(self, split: str):
        """
        获取指定split的DataLoader
        
        Args:
            split: 'train', 'validation', 或 'test'
            
        Returns:
            DataLoader对象
        """
        if split not in self.dataloaders:
            raise ValueError(f"未找到{split}数据集")
        return self.dataloaders[split]
    
    def get_dataset(self, split: str):
        """获取指定split的Dataset"""
        if split not in self.datasets:
            raise ValueError(f"未找到{split}数据集")
        return self.datasets[split]
    
    def get_train_dataloader(self):
        """获取训练DataLoader"""
        return self.get_dataloader('train')
    
    def get_val_dataloader(self):
        """获取验证DataLoader"""
        return self.get_dataloader('validation')
    
    def get_test_dataloader(self):
        """获取测试DataLoader"""
        return self.get_dataloader('test')


# 示例用法
if __name__ == "__main__":
    # 示例1: VQA任务
    # pipeline = DataPipeline('config/vqa_config.yaml')
    # pipeline.setup()
    # train_loader = pipeline.get_train_dataloader()
    # for batch in train_loader:
    #     print("Batch keys:", batch.keys())
    #     print("Image shape:", batch['pixel_values'].shape)
    #     print("Question shape:", batch['input_ids'].shape)
    #     print("Answer shape:", batch['labels'].shape)
    #     break
    
    # 示例2: Image Captioning任务
    # pipeline = DataPipeline('config/caption_config.yaml')
    # pipeline.setup()
    # train_loader = pipeline.get_train_dataloader()
    # for batch in train_loader:
    #     print("Batch keys:", batch.keys())
    #     print("Image shape:", batch['pixel_values'].shape)
    #     print("Caption shape:", batch['labels'].shape)
    #     break
    
    # 示例3: 文本分类任务（向后兼容）
    # pipeline = DataPipeline('config/text_classification_config.yaml')
    # pipeline.setup()
    
    print("DataPipeline模块加载完成 - 支持VQA、Image Captioning等多种任务")
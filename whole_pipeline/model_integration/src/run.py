from pipeline import create_pipeline

# 创建管线
pipeline = create_pipeline('/home/zhuxuzhou/VQA_Auto/whole_pipeline/model_integration/config/vqa_config.yaml', experiment_name='my_experiment')

# 设置并训练
pipeline.setup_data()
pipeline.setup_model()
pipeline.setup_training()
pipeline.train()

# # 评估
# results = pipeline.evaluate(split='validation')
# print(f"验证准确率: {results['accuracy']:.4f}")

# # 预测
# answer = pipeline.predict('path/to/image.jpg', 'What color is the car?')
# print(f"答案: {answer}")

# 保存
pipeline.save()
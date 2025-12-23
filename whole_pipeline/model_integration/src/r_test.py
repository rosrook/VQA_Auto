import torch
from PIL import Image
from pathlib import Path
from transformers import (
    AutoModelForVisualQuestionAnswering,
    AutoProcessor
)

if __name__=="__main__":
    checkpoint_path = "/home/zhuxuzhou/VQA_Auto/whole_pipeline/model_integration/models_after_training"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 直接加载模型和processor
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        checkpoint_path
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    model.eval()
    
    # 加载图像和问题
    image_path = "/home/zhuxuzhou/VQA_Auto/whole_pipeline/model_integration/src/r_test.jpeg"  # 替换为实际图像路径
    question = "Introduce the picture detailed."
    
    image = Image.open(image_path).convert('RGB')
    
    # 处理输入
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    # 生成答案
    with torch.no_grad():
        out = model.generate(**inputs, max_length=20, num_beams=3)
    
    # 解码答案
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"问题: {question}")
    print(f"答案: {answer}")
    print()
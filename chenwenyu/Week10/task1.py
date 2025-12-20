import os
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import time

def analyze_images_in_folder(folder_path="./pics"):
    """
    批量分析指定文件夹中的所有图片
    """
    # 检查文件夹是否存在
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误：文件夹 {folder_path} 不存在")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f"1_*{ext}"))
        image_files.extend(folder.glob(f"1_*{ext.upper()}"))
    
    if not image_files:
        print(f"在 {folder_path} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始加载模型...")
    
    # 加载模型和处理器
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "../../../models/Qwen/Qwen2-VL-7B-Instruct",
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "../../../models/Qwen/Qwen2-VL-7B-Instruct",
            trust_remote_code=True
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 分析每张图片
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n{'='*50}")
        print(f"分析第 {i} 张图片: {image_path.name}")
        print(f"{'='*50}")
        
        try:
            # 打开图片
            image = Image.open(image_path)
            
            # 构建分析请求
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "请详细分析这张图片中的主要物体、场景和内容。包括：1.主要物体是什么 2.场景类型 3.颜色和氛围 4.其他显著特征"}
                    ]
                }
            ]
            
            # 处理输入
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            # 生成分析结果
            print("正在分析中...")
            start_time = time.time()
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
            
            # 解码结果
            generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
            analysis_result = processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )
            
            end_time = time.time()
            print(f"分析完成 (耗时: {end_time - start_time:.2f}秒)")
            print(f"\n分析结果:\n{analysis_result}")
            
        except Exception as e:
            print(f"分析图片 {image_path.name} 时出错: {e}")
            continue
        
        print(f"\n{'-'*50}")

def main():
    """
    主函数
    """
    print("🚀 Qwen2-VL 图片分析工具")
    print("开始分析 ./pics 目录中的图片...")
    
    # 分析图片
    analyze_images_in_folder("./pics")
    
    print("\n🎉 所有图片分析完成！")

if __name__ == "__main__":
    main()

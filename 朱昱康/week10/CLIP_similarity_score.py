from PIL import Image
import os
import torch
from modelscope import ChineseCLIPProcessor, ChineseCLIPModel

# 加载CLIP模型和处理器
model = ChineseCLIPModel.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")

# 图片路径，需要替换
img_dir = "/Users/Yukang/Documents/LLM/week10/第10周：多模态大模型/Week10/images" 

# 文本标签
texts = ["皮卡丘", "小火龙", "杰尼龟", "妙蛙种子", "小拳石", "双弹瓦斯", "可达鸭", "超梦", "蚊香蝌蚪", "喵喵"]

# 获取所有图片文件
image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg') or f.endswith('.jpg')])

# 存储结果
results = []

print("开始计算图片与文本的相似度分数...")
print("=" * 50)

# 对每张图片进行处理
for img_file in image_files:
    # 加载图片
    img_path = os.path.join(img_dir, img_file)
    image = Image.open(img_path)
    
    # 计算图文相似度
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图文相似度分数
    probs = logits_per_image.softmax(dim=1)  # 转换为概率
    
    # 获取最匹配的文本标签
    max_prob_idx = torch.argmax(probs, dim=1).item()
    max_prob = probs[0, max_prob_idx].item()
    best_match_text = texts[max_prob_idx]
    
    # 保存结果
    results.append({
        'image': img_file,
        'best_match': best_match_text,
        'probability': max_prob,
        'all_probs': {texts[i]: probs[0, i].item() for i in range(len(texts))}
    })
    
    # 打印结果
    print(f"图片: {img_file}")
    print(f"最匹配的文本: {best_match_text} (概率: {max_prob:.4f})")
    print("-" * 30)

# 打印所有结果的汇总
print("\n所有图片的匹配结果:")
print("=" * 50)
for result in results:
    print(f"{result['image']} -> {result['best_match']} (概率: {result['probability']:.4f})")

# 也可以查看每张图片与所有文本的相似度分数
print("\n详细相似度分数:")
print("=" * 50)
for result in results:
    print(f"\n图片: {result['image']}")
    sorted_probs = sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True)
    for text, prob in sorted_probs:
        print(f"  {text}: {prob:.4f}")
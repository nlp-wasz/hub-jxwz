import glob, json, os
from PIL import Image
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import jieba


def load_local_clip_model(model_path="D:\learning\八斗\Week10\model\chinese-clip-vit-base-patch16"):
    """
    加载本地中文CLIP模型

    Args:
        model_path (str): 本地模型路径

    Returns:
        tuple: (model, processor)
    """
    print("正在加载本地中文CLIP模型...")
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    print("模型加载完成")
    return model, processor


def extract_image_features_local(image_paths, model, processor, batch_size=10):
    """
    使用本地模型提取图像特征

    Args:
        image_paths (list): 图像路径列表
        model: CLIP模型
        processor: 处理器
        batch_size (int): 批处理大小

    Returns:
        np.array: 图像特征向量
    """
    print("正在提取图像特征...")
    image_features_list = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"处理批次 {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")

        # 加载图像
        imgs = [Image.open(path) for path in batch_paths]

        # 预处理
        inputs = processor(images=imgs, return_tensors="pt", padding=True)

        # 提取特征
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features.cpu().numpy()
            image_features_list.append(image_features)

    # 合并所有批次的特征
    image_features = np.vstack(image_features_list)
    # 归一化
    image_features = normalize(image_features)
    print(f"图像特征提取完成，特征维度: {image_features.shape}")

    return image_features


def extract_text_features_local(texts, model, processor, batch_size=10):
    """
    使用本地模型提取文本特征

    Args:
        texts (list): 文本列表
        model: CLIP模型
        processor: 处理器
        batch_size (int): 批处理大小

    Returns:
        np.array: 文本特征向量
    """
    print("正在提取文本特征...")
    text_features_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"处理批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

        # 预处理
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)

        # 提取特征
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features.cpu().numpy()
            text_features_list.append(text_features)

    # 合并所有批次的特征
    text_features = np.vstack(text_features_list)
    # 归一化
    text_features = normalize(text_features)
    print(f"文本特征提取完成，特征维度: {text_features.shape}")

    return text_features


def calculate_similarity_matrix(image_features, text_features):
    """
    计算图像和文本特征的相似度矩阵

    Args:
        image_features (np.array): 图像特征向量
        text_features (np.array): 文本特征向量

    Returns:
        np.array: 相似度矩阵
    """
    # 计算余弦相似度矩阵
    similarity_matrix = np.dot(text_features, image_features.T)
    return similarity_matrix


def match_image_text_pairs_local(image_paths, texts, model, processor):
    """
    使用本地模型执行图文匹配任务

    Args:
        image_paths (list): 图像路径列表
        texts (list): 文本列表
        model: CLIP模型
        processor: 处理器

    Returns:
        list: 匹配结果
    """
    # 提取图像特征
    image_features = extract_image_features_local(image_paths, model, processor)

    # 提取文本特征
    text_features = extract_text_features_local(texts, model, processor)

    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(image_features, text_features)

    # 生成匹配结果
    results = []
    for i, text in enumerate(texts):
        for j, image_path in enumerate(image_paths):
            similarity = similarity_matrix[i, j]
            results.append({
                "text_index": i,
                "image_index": j,
                "text": text,
                "image_path": image_path,
                "similarity": float(similarity)
            })

    return results


def visualize_matching_results(image_paths, texts, results, top_k=3):
    """
    可视化匹配结果

    Args:
        image_paths (list): 图像路径列表
        texts (list): 文本列表
        results (list): 匹配结果
        top_k (int): 显示前K个匹配结果
    """
    # 按文本分组结果
    text_groups = {}
    for result in results:
        text_idx = result["text_index"]
        if text_idx not in text_groups:
            text_groups[text_idx] = []
        text_groups[text_idx].append(result)

    # 对每组结果按相似度排序
    for text_idx in text_groups:
        text_groups[text_idx].sort(key=lambda x: x["similarity"], reverse=True)

    # 显示结果
    for text_idx, group_results in text_groups.items():
        print(f"\n文本: {texts[text_idx]}")
        print("匹配的图像:")

        # 显示前top_k个匹配结果
        for i, result in enumerate(group_results[:top_k]):
            print(f"  {i + 1}. {result['image_path']} (相似度: {result['similarity']:.4f})")

            # 显示图像
            plt.figure(figsize=(5, 5))
            img = Image.open(result['image_path'])
            plt.imshow(img)
            plt.title(f"相似度: {result['similarity']:.4f}")
            plt.axis('off')
            plt.show()


# 使用示例
if __name__ == "__main__":
    # 准备数据
    # 本地图像路径列表（10个图像）
    local_image_paths = [
        "D:\learning\photo\downloaded-image1.png",
        "D:\learning\photo\downloaded-image2.png",
        "D:\learning\photo\downloaded-image3.png",
        "D:\learning\photo\downloaded-image4.png",
        "D:\learning\photo\downloaded-image5.png",
        "D:\learning\photo\downloaded-image6.png",
        "D:\learning\photo\downloaded-image7.png",
        "D:\learning\photo\downloaded-image8.png",
        "D:\learning\photo\downloaded-image9.png",
        "D:\learning\photo\downloaded-image10.png",
    ]

    # 文本列表（10个文本）
    texts = [
        "一只可爱的小猫在阳光下打盹",
        "一群孩子在公园里踢足球",
        "美丽的山水风景画",
        "现代城市夜景",
        "美味的食物摆盘",
        "科技感十足的机器人",
        "古老的建筑群",
        "海洋中的珊瑚礁",
        "森林里的小动物",
        "运动场上的运动员"
    ]

    try:
        # 加载本地模型
        model, processor = load_local_clip_model()

        # 执行图文匹配
        print("开始执行图文匹配任务...")
        results = match_image_text_pairs_local(local_image_paths, texts, model, processor)

        # 按相似度排序并显示结果
        results.sort(key=lambda x: x["similarity"], reverse=True)

        print("\n=== 图文匹配结果 ===")
        for i, result in enumerate(results[:10]):  # 显示前10个最佳匹配
            print(f"{i + 1}. 相似度: {result['similarity']:.4f}")
            print(f"   图像: {result['image_path']}")
            print(f"   文本: {result['text']}")
            print()

        # 可视化匹配结果
        visualize_matching_results(local_image_paths, texts, results)

    except Exception as e:
        print(f"执行图文匹配时出错: {e}")

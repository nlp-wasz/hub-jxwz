import glob
import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import jieba
from datasets import load_dataset


# 加载数据集
def load_imagenet_dataset():
    """加载并预处理ImageNet样本数据集"""
    print("正在加载Multimodal-Fatima/ImagenetZ1k_sample_validation数据集...")
    dataset = load_dataset("Multimodal-Fatima/Imagenet1k_sample_validation")

    # 查看数据集结构
    print("\n数据集信息:")
    print(f"列名: {dataset['validation'].column_names}")
    print(f"样本数: {len(dataset['validation'])}")

    # 查看一个样本示例
    sample = dataset['validation'][0]
    print(f"\n样本示例:")
    print(f"图像类型: {type(sample['image'])}")
    print(f"标签ID: {sample['label']}")
    print(f"词汇表: {sample['lexicon']}")

    return dataset


# 主函数
def main():
    # 加载数据集
    dataset = load_imagenet_dataset()
    validation_data = dataset['validation']

    # 准备图像路径和标题
    img_paths = []
    img_captions = []

    # 创建临时目录保存图像
    os.makedirs('./temp_images', exist_ok=True)

    print("\n准备图像和标题数据...")
    for i, item in enumerate(tqdm(validation_data)):
        # 保存图像到临时文件
        img_path = f'./temp_images/image_{i}.jpg'
        item['image'].save(img_path)
        img_paths.append(img_path)

        # 使用词汇表中的第一个词作为标题
        caption = item['lexicon'][0] if item['lexicon'] else "未知类别"
        img_captions.append(caption)

    # 加载模型
    print("\n加载ChineseCLIP模型...")
    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    # 提取图像特征
    print("\n提取图像特征...")
    img_image_feat = []
    batch_size = 20

    for idx in tqdm(range(len(img_paths) // batch_size + 1)):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, len(img_paths))

        if start_idx >= end_idx:
            break

        # 加载图像批次
        imgs = [Image.open(path) for path in img_paths[start_idx:end_idx]]

        # 处理图像
        inputs = processor(images=imgs, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features.cpu().numpy()
            img_image_feat.append(image_features)

    img_image_feat = np.vstack(img_image_feat)
    img_image_feat = normalize(img_image_feat)
    print(f"图像特征形状: {img_image_feat.shape}")

    # 提取文本特征
    print("\n提取文本特征...")
    img_texts_feat = []

    for idx in tqdm(range(len(img_captions) // batch_size + 1)):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, len(img_captions))

        if start_idx >= end_idx:
            break

        # 获取文本批次
        texts = img_captions[start_idx:end_idx]

        # 处理文本
        inputs = processor(text=texts, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features.cpu().numpy()
            img_texts_feat.append(text_features)

    img_texts_feat = np.vstack(img_texts_feat)
    img_texts_feat = normalize(img_texts_feat)
    print(f"文本特征形状: {img_texts_feat.shape}")

    # 文本到图像检索示例
    print("\n文本到图像检索示例...")
    query_idx = 10  # 选择一个查询文本

    # 计算相似度
    sim_result = np.dot(img_texts_feat[query_idx], img_image_feat.T)
    sim_idx = sim_result.argsort()[::-1][1:4]  # 取最相似的3个图像（排除自身）

    print(f'输入文本: "{img_captions[query_idx]}"')

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'文本查询: "{img_captions[query_idx]}"', fontsize=16)

    for i, idx in enumerate(sim_idx):
        plt.subplot(1, 3, i + 1)
        plt.imshow(Image.open(img_paths[idx]))
        plt.title(f'相似度: {sim_result[idx]:.3f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 图像到文本检索示例
    print("\n图像到文本检索示例...")
    query_idx = 15  # 选择一个查询图像

    # 显示查询图像
    plt.figure(figsize=(5, 5))
    plt.imshow(Image.open(img_paths[query_idx]))
    plt.title(f'查询图像: {img_captions[query_idx]}')
    plt.axis('off')
    plt.show()

    # 计算相似度
    sim_result = np.dot(img_image_feat[query_idx], img_texts_feat.T)
    sim_idx = sim_result.argsort()[::-1][1:4]  # 取最相似的3个文本（排除自身）

    print(f'最相似的文本描述:')
    for i, idx in enumerate(sim_idx):
        print(f"{i + 1}. {img_captions[idx]} (相似度: {sim_result[idx]:.3f})")

    # 分词和词级特征提取
    print("\n分词和词级特征提取...")
    img_captions2words = [jieba.lcut(caption) for caption in img_captions]
    all_words = sum(img_captions2words, [])
    unique_words = list(set(all_words))
    print(f"总词汇数: {len(all_words)}, 唯一词汇数: {len(unique_words)}")

    # 提取词级特征
    print("\n提取词级特征...")
    word_features = []
    word_batch_size = 100

    for idx in tqdm(range(len(unique_words) // word_batch_size + 1)):
        start_idx = idx * word_batch_size
        end_idx = min((idx + 1) * word_batch_size, len(unique_words))

        if start_idx >= end_idx:
            break

        # 获取词汇批次
        words = unique_words[start_idx:end_idx]

        # 处理文本
        inputs = processor(text=words, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features.cpu().numpy()
            word_features.append(text_features)

    word_features = np.vstack(word_features)
    word_features = normalize(word_features)
    print(f"词级特征形状: {word_features.shape}")

    # 创建词汇到特征的映射
    word_to_feature = {word: feature for word, feature in zip(unique_words, word_features)}

    # 图像到词汇检索示例
    print("\n图像到词汇检索示例...")
    query_idx = 20  # 选择一个查询图像

    # 显示查询图像
    plt.figure(figsize=(5, 5))
    plt.imshow(Image.open(img_paths[query_idx]))
    plt.title(f'查询图像: {img_captions[query_idx]}')
    plt.axis('off')
    plt.show()

    # 计算图像特征与所有词汇的相似度
    image_feature = img_image_feat[query_idx]
    word_similarities = np.dot(image_feature, word_features.T)

    # 获取最相似的前10个词汇
    top_word_indices = word_similarities.argsort()[::-1][:10]
    top_words = [unique_words[i] for i in top_word_indices]
    top_similarities = word_similarities[top_word_indices]

    print(f"与图像最相关的词汇:")
    for word, sim in zip(top_words, top_similarities):
        print(f"{word}: {sim:.3f}")

    # 清理临时文件
    print("\n清理临时文件...")
    for path in img_paths:
        os.remove(path)
    os.rmdir('./temp_images')

    print("\n处理完成!")


if __name__ == "__main__":
    main()
# CLIP 模型（通过对比学习，可以进行零样本的图文匹配
# 通过将图片和文本进行编码，之后计算预先相似度，得到正样本和负样本）

import glob  # 加载文件地址信息（返回 list）
import json  # 处理json信息
import os, torch, tqdm, jieba, matplotlib.pyplot as plt, numpy as np

from sklearn.preprocessing import normalize
from PIL import Image  # 加载图像
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CosineSimilarity
from torch.optim import SGD

device = "cuda" if torch.cuda.is_available() else "cpu"


# 1.加载图片地址 和 图片信息（caption.json 描述信息）
def load_images_caption():
    # 使用 glob 加载图片地址信息
    img_paths = glob.glob("./data/img/*.jpg")
    img_paths = sorted(img_paths)
    print(img_paths[:2])

    # 使用 json 加载caption.json 图片描述信息
    img_captions = json.load(open("./data/img/caption.json", encoding="utf-8"))
    print(img_captions[:2])

    return img_paths, img_captions


# 2.提取 所有图片的 路径名称和对应的描述信息
def extract_image_name_caption(img_paths, img_captions):
    # 1.从 img_paths 中提取文件名称（包含后缀）
    img_names = [os.path.basename(img_path) for img_path in img_paths]

    # 2.从 img_captions 中提取 image_id、caption 组成键值对
    img_id_captions = {img_caption["image_id"]: img_caption["caption"][0] for img_caption in img_captions}
    captions = [img_id_captions[os.path.basename(img_path)] for img_path in img_paths]

    # 3.使用 Image 加载图片，并展示
    # img = Image.open(img_paths[0])
    # img.show()

    return img_names, img_id_captions, captions


# 3.加载CLIP模型
def load_clip_preprocessor_model():
    # 加载 中文 CLIP 处理器（使用 transformers.ChineseCLIPProcessor）
    # 内置了 image_processor处理器 和 tokenizer文本分词处理器
    clip_processor = ChineseCLIPProcessor.from_pretrained("../models/clip/chinese-clip-vit-base-patch16")

    # 加载 中文 CLIP 模型（使用 transformers.ChineseCLIPModel）
    clip_model = ChineseCLIPModel.from_pretrained("../models/clip/chinese-clip-vit-base-patch16")
    return clip_processor, clip_model


# 4.获取图片特征（图像编码） 和 描述文本特征（文本编码）
def image_text_encoder(img_paths, img_id_captions, clip_processor, clip_model):
    clip_model.eval()

    # 1.图像编码
    img_features = []
    for idx in tqdm.tqdm(range(len(img_paths) // 2)):
        # ①使用 PIL.Image 加载图像
        img_Image = [Image.open(img_path).convert("RGB") for img_path in img_paths[idx * 2: idx * 2 + 2]]

        # ② 使用 clip_processor 对图像进行编码处理
        img_encoder = clip_processor(images=img_Image, return_tensors="pt")

        # ③ 使用 clip_model 获取图像特征
        with torch.no_grad():
            img_feature = clip_model.get_image_features(**img_encoder)

        # ④ 转换为 numpy
        img_features.append(img_feature.data.numpy())
    img_features = np.vstack(img_features)
    # normalize  归一化处理，便于计算 余弦相似度
    img_features = normalize(img_features)

    # 2.文本编码
    text_features = []
    for idx in tqdm.tqdm(range(len(img_paths) // 2)):
        # ① 获取 图片对应的 描述信息
        img_captions = [img_id_captions[os.path.basename(img_path)] for img_path in img_paths[idx * 2: idx * 2 + 2]]

        # ② 使用 clip_processor 对文本进行处理
        text_encoder = clip_processor(text=img_captions, return_tensors="pt", padding=True)

        # ③ 使用 clip_model 获取文本特征
        text_feature = clip_model.get_text_features(**text_encoder)

        # ④转换为 numpy
        text_features.append(text_feature.data.numpy())
    text_features = np.vstack(text_features)
    # normalize  归一化处理，便于计算 余弦相似度
    text_features = normalize(text_features)

    return img_features, text_features


# 5.通过 计算余弦相似度 判断（与图片最相似的文本，与文本最相似的图片）
def images_texts_cosine_similiarity(img_paths, captions, img_features, text_features):
    # 1.获取 与图片最相似的文本（TOP3）
    for idx, img_feature in enumerate(img_features):
        # 余弦相似度
        similar = np.dot(np.expand_dims(img_feature, axis=0), text_features.T)

        # 排序，获取 TOP3
        # image_texts_similiarity = image_texts_similiarity.argsort(-1)
        similar = similar.flatten().argsort(-1)[::-1][:3]

        # 图片 可视化
        plt.figure(figsize=(12, 5))
        plt.imshow(Image.open(img_paths[idx]))
        plt.show()

        simia_text = [captions[i] for i in similar]
        print(simia_text)

        break

    # 2.获取 与文本最相似的图片（TOP3）
    for idx, text_feature in enumerate(text_features):
        # 余弦相似度
        similar = np.dot(text_feature, img_features.T)

        # 排序，获取 TOP3
        similar = similar.flatten().argsort(-1)[::-1][:3]

        # 图片 可视化
        plt.figure(figsize=(12, 5))
        plt.subplot(131)
        plt.imshow(Image.open(img_paths[similar[0]]))

        plt.subplot(132)
        plt.imshow(Image.open(img_paths[1]))

        plt.subplot(133)
        plt.imshow(Image.open(img_paths[2]))
        plt.show()

        print(f"文本：{captions[idx]}")
        break


# 6.将文本通过 jieba 进行分词，计算 与图片相似的标签TOP10
def images_labels_cosine_similarity(img_paths, captions, img_features, clip_processor, clip_model):
    clip_model.eval()

    # 1.对文本进行 jieba 分词
    text_jieba = [word for caption in captions for word in jieba.cut(caption) if len(word) > 1]
    text_jieba = list(set(text_jieba))

    # 2.对 text_jieba 进行processor处理
    text_encoder = clip_processor(text=text_jieba, return_tensors="pt", padding=True)

    # 3.CLIP模型 获取文本标签特征
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_encoder)
        text_features = normalize(text_features.data.numpy())

    # 4.计算 每一张图片 与 文本标签 的余弦相似度
    for idx, img_feature in enumerate(img_features):
        # normalize 处理后，点积就是余弦相似度
        similar = np.dot(img_feature, text_features.T)

        # 获取 TOP10
        similar = similar.flatten().argsort(-1)[::-1][:3]

        # 图片可视化
        plt.figure(figsize=(12, 5))
        plt.imshow(Image.open(img_paths[idx]))
        plt.show()

        print(f"标签：{[text_jieba[i] for i in similar]}")


# 7.CLIP 原始模型 余弦相似度均值 计算
def cosine_similiarity(img_features, text_features):
    # 计算 原始特征  图文匹配之间的相似度 准确性均值
    for i in range(3):
        similar = np.dot(img_features, text_features.T)
        similar_max = similar.max(axis=-1)
        similar_mean = similar_max.mean(axis=-1)
        print(f"CLIP模型训练前相似度均值：{similar_mean}")


# 8.CLIP 模型微调
def clip_model_train(img_paths, img_captions, clip_processor, clip_model):
    clip_model.load_state_dict(torch.load("./data/clip_model.pt"))
    clip_model.eval()

    # CLIP 模型训练
    # ①构建 Dataset、DataLoader、损失函数、优化器
    clip_dataset = CLIPDataset(img_paths, img_captions, clip_processor)
    clip_dataloader = DataLoader(dataset=clip_dataset, batch_size=3, shuffle=True)

    loss_func = CosineSimilarity()
    optimizer = SGD(params=clip_model.parameters(), lr=0.0001)

    # ②模型训练
    clip_model.train()
    clip_model.to(device)
    for epoch in range(100):
        for batch_idx, (pixel_values, input_ids, attention_mask) in enumerate(clip_dataloader):
            img_features = clip_model.get_image_features(pixel_values=pixel_values)
            text_features = clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

            # 计算余弦相似度 均值（均值越大越好，损失值越小越好）
            loss = loss_func(img_features, text_features)
            loss = 1 - loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"第 {epoch} 次循环，第 {batch_idx} 批次，损失值：{loss}")

    # ③保存模型
    torch.save(clip_model.state_dict(), "./data/clip_model.pt")


class CLIPDataset(Dataset):
    def __init__(self, img_paths, img_captions, clip_processor):
        super(CLIPDataset, self).__init__()
        self.img_paths = img_paths
        self.img_captions = img_captions
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_Image = Image.open(self.img_paths[idx])
        inputs = self.clip_processor(
            images=img_Image,
            text=self.img_captions[idx],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt")

        inputs_encoder = {key: value.squeeze().to(device) for key, value in inputs.items()}

        return inputs_encoder["pixel_values"], inputs_encoder["input_ids"], inputs_encoder["attention_mask"]


# 9.使用 训练后的模型 重新对图文进行编码，提取特征
def train_model_after(img_paths, captions, clip_processor, clip_model):
    clip_model.load_state_dict(torch.load("./data/clip_model.pt"))
    clip_model.eval()
    clip_model.to("cpu")

    # 图文编码 编码处理
    img_Images = [Image.open(img_path) for img_path in img_paths]
    image_encoder = clip_processor(images=img_Images, return_tensors="pt")
    text_encoder = clip_processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # 模型提取特征
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_encoder)
        img_features = clip_model.get_image_features(**image_encoder)

        text_features = normalize(text_features.data.numpy())
        img_features = normalize(img_features.data.numpy())

    # 计算相似度
    for i in range(3):
        similar = np.dot(img_features, text_features.T).max(-1).mean()
        print(f"模型训练后相似度均值：{similar}")


if __name__ == "__main__":
    # 1.加载图片地址 和 图片信息（caption.json 描述信息）
    img_paths, img_captions = load_images_caption()

    # 2.提取 所有图片的 路径名称和对应的描述信息
    img_names, img_id_captions, captions = extract_image_name_caption(img_paths, img_captions)

    # 3.加载CLIP模型
    clip_processor, clip_model = load_clip_preprocessor_model()

    # 4.获取图片特征（图像编码） 和 描述文本特征（文本编码）
    img_features, text_features = image_text_encoder(img_paths, img_id_captions, clip_processor, clip_model)

    # 5.通过 计算余弦相似度 判断（与图片最相似的文本，与文本最相似的图片）
    # images_texts_cosine_similiarity(img_paths, captions, img_features, text_features)

    # 6.将文本通过 jieba 进行分词，计算 与图片相似的标签TOP10
    # images_labels_cosine_similarity(img_paths, captions, img_features, clip_processor, clip_model)

    # 7.CLIP 原始模型 余弦相似度均值 计算
    cosine_similiarity(img_features, text_features)

    # 8.CLIP 模型微调
    # clip_model_train(img_paths, captions, clip_processor, clip_model)

    # 9.使用 训练后的模型 重新对图文进行编码，提取特征
    train_model_after(img_paths, captions, clip_processor, clip_model)

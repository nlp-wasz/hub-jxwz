import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ============================================================
# 1️⃣ 配置路径
# ============================================================
DATA_DIR = "../flickr8k/Flicker8k_Dataset"
CAPTION_FILE = "../flickr8k/Flickr8k.token.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# ============================================================
# 2️⃣ 解析 Flickr8k 数据
# ============================================================
class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, caption_file, processor, max_samples=5000):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []

        with open(caption_file, "r") as f:
            for line in f:
                img_id, caption = line.strip().split("\t")
                img_name = img_id.split("#")[0]
                img_path = os.path.join(data_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, caption))
                if len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return {"image": image, "text": caption}

# ============================================================
# 3️⃣ 初始化模型和优化器
# ============================================================
model_name = "../../../models/AI-ModelScope/chinese-clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

optimizer = AdamW(model.parameters(), lr=1e-5)
scaler = GradScaler(device="cuda")

# ============================================================
# 4️⃣ 创建 DataLoader
# ============================================================
dataset = Flickr8kDataset(DATA_DIR, CAPTION_FILE, processor)
print(f"✅ 样本数量: {len(dataset)}")

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)    #删除CLIP Model不需要的字段
    return inputs

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ============================================================
# 5️⃣ 训练（仅示例1轮）
# ============================================================
model.train()
for epoch in range(1):
    for batch in tqdm(dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs,return_loss=True)
            loss = outputs.loss

        #print(type(outputs))
        #print(outputs.keys())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch done. Loss={loss.item():.4f}")

# ============================================================
# 6️⃣ 保存模型
# ============================================================
model.save_pretrained("./clip_finetuned_flickr8k")
processor.save_pretrained("./clip_finetuned_flickr8k")
print("✅ 模型保存到 ./clip_finetuned_flickr8k")

# ============================================================
# 7️⃣ 推理示例
# ============================================================
model.eval()
texts = ["A man playing guitar", "A dog running in the park", "A group of people sitting on grass"]

# 选取一部分图片（例如前 100 张）
num_images = 100
images = [Image.open(dataset.samples[i][0]).convert("RGB") for i in range(num_images)]
image_paths = [dataset.samples[i][0] for i in range(num_images)]

# 获取文本与图片特征
with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
    # 1️⃣ 文本特征
    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
    text_inputs.pop("token_type_ids", None)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = model.get_text_features(**text_inputs)

    # 2️⃣ 图像特征
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    image_inputs.pop("token_type_ids", None)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    image_features = model.get_image_features(**image_inputs)

# 特征归一化（CLIP 默认对比相似度计算前要 normalize）
text_features = nn.functional.normalize(text_features, dim=-1)
image_features = nn.functional.normalize(image_features, dim=-1)

# 相似度矩阵： [num_texts, num_images]
similarity = text_features @ image_features.T

# 输出每个文本最匹配的图片
for i, text in enumerate(texts):
    best_img_idx = similarity[i].argmax().item()
    best_img_path = image_paths[best_img_idx]
    print(f"📝 文本: '{text}'")
    print(f"🏞️ 最相似图片: {best_img_path}")
    print(f"🔢 相似度: {similarity[i, best_img_idx]:.4f}")
    print("-" * 60)

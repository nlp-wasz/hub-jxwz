import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 1. 路径
# best_model_path = "./results/best_model"
best_model_path = "./results/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载 tokenizer 和模型（一次即可）
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)
model.eval()

# 3. 标签映射（根据你训练时的 LabelEncoder 顺序）
id2label = {0: "negative", 1: "positive"}   # 按需修改

def predict(text: str, top_k=1):
    """
    输入文本，返回预测标签与概率
    :param text: 外卖评价文本
    :param top_k: 返回前 k 个最可能的标签
    :return: [(label, prob), ...]
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()   # shape: [num_labels]
        probs = torch.softmax(logits, dim=-1)

    # 取 top-k
    top_probs, top_indices = torch.topk(probs, k=top_k)
    results = [(id2label[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
    return results

# ---------------------------
# 用法示例
if __name__ == "__main__":
    text = "这家外卖真的又快又好吃！"
    print(predict(text))
    # 输出示例：[('positive', 0.9876)]
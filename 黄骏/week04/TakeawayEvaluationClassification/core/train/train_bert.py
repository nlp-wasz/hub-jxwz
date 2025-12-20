import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from modelscope.hub.snapshot_download import snapshot_download

# 加载和预处理数据
dataset = pd.read_csv("../../assets/dataset/takeaway_10k.csv")

# 清洗数据
dataset = dataset.dropna(subset=["review", "label"]).copy()
dataset["review"] = dataset["review"].astype(str)
dataset["label"] = dataset["label"].astype(int)

texts = dataset["review"].values
labels = dataset["label"].values
max_length = round(dataset["review"].str.len().quantile(0.99)) # 99%分位数作为最大序列长度

# 分割数据为训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    texts,                  # 文本数据
    labels,                 # 数字标签
    test_size=0.2,          # 测试集20%比例
    stratify=labels,        # 确保训练集和测试集标签分布一致
)

model_path = "../../assets/models/google-bert/bert-base-chinese"

# 检查模型是否已经存在，如果不存在则下载
if not os.path.exists(model_path):
    print("模型不存在，正在从ModelScope下载")
    # 使用ModelScope下载模型
    snapshot_download(model_id='google-bert/bert-base-chinese', cache_dir="../../assets/models")
    print("模型下载完成")
else:
    print("本地模型已存在，将直接加载")

# 从本地预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 使用分词器对训练集测试集文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
train_encodings = tokenizer(train_x.tolist(), truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_x.tolist(), truncation=True, padding=True, max_length=max_length)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],              # 文本token ID
    'attention_mask': train_encodings['attention_mask'],    # 注意力掩码
    'labels': train_y                                       # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_y
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)    # axis=-1 最后一个维度
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

data_seed = 42

# 配置训练参数
train_args = TrainingArguments(
    output_dir='../../assets/weights/bert/',    # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                         # 训练的总轮数
    learning_rate=2e-5,                         # 学习率
    lr_scheduler_type="cosine",                 # 学习率调度器 liner cosine polynomial
    per_device_train_batch_size=16,             # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=32,              # 评估时每个设备的批次大小
    gradient_accumulation_steps=2,              # 梯度累积步数
    warmup_ratio=0.1,                           # 预热率
    weight_decay=0.01,                          # 权重衰减，用于防止过拟合
    logging_dir='./logs',                       # 日志存储目录
    logging_steps=100,                          # 每隔100步 记录一次日志
    eval_strategy="epoch",                      # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",                      # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,                # 训练结束后加载效果最好的模型
    # fp16=True,                                  # 使用混合精度训练
    gradient_checkpointing=True,                # 梯度检查点，减少内存占用
    seed=data_seed,                             # 随机种子
)

# 实例化 Trainer
trainer = Trainer(
    model=model,                        # 训练的模型
    args=train_args,                    # 训练参数
    train_dataset=train_dataset,        # 训练集
    eval_dataset=test_dataset,          # 评估测试集
    compute_metrics=compute_metrics     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在评估测试集上进行评估
trainer.evaluate()

# 保存最优模型
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"最优模型位于：{best_model_path}")
    torch.save(best_model.state_dict(), "../../assets/weights/bert.pt")
    print("最优模型保存在 assets/weights/bert.pt")
else:
    print("未找到最优模型检查点")
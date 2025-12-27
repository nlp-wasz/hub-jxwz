#!/usr/bin/env python3
"""
简化版BERT微调脚本
快速训练模型用于测试
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
print("加载数据...")
df = pd.read_csv("./assets/dataset/作业数据-waimai_10k.csv")
print(f"数据集大小: {len(df)}")
print(f"类别分布: {df['label'].value_counts().to_dict()}")

# 限制数据量用于快速测试（可选）
# df = df.sample(n=2000, random_state=42)

# 分割数据
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

print(f"训练集: {len(train_texts)}, 测试集: {len(test_texts)}")

# 初始化分词器和模型
print("初始化模型...")
model_path = "./assets/models/google-bert/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 编码数据
def encode_data(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )
    
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

print("编码数据...")
train_dataset = encode_data(train_texts, train_labels)
test_dataset = encode_data(test_texts, test_labels)

# 训练参数（简化版）
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

print("开始训练...")
trainer.train()

# 评估
results = trainer.evaluate()
print(f"测试结果: {results}")

# 保存模型
print("保存模型...")
model.save_pretrained("./models/fine_tuned_bert_simple")
tokenizer.save_pretrained("./models/fine_tuned_bert_simple")

print("训练完成！模型已保存到 ./models/fine_tuned_bert_simple")
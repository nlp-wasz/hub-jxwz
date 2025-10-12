import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_metric
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU，如果有多个GPU可以指定

# # 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# if torch.cuda.is_available():
#     print(f"GPU名称: {torch.cuda.get_device_name(0)}")
#     print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 加载数据
df = pd.read_csv('./作业数据-waimai_10k.csv')

# 数据预处理
df = df.dropna()
df['label'] = df['label'].astype(int)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 数据编码函数
def preprocess_function(examples):
    return tokenizer(
        examples['review'],
        truncation=True,
        max_length=128,
        padding=False
    )

# 转换为Dataset格式
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 对数据集进行tokenize
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# 定义评估指标
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 创建数据收集器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate(test_dataset)
print(f"测试集准确率: {eval_results['eval_accuracy']}")

# 保存模型
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import os
import torch
import joblib

# 确保输出目录存在
os.makedirs('../homework/bert/', exist_ok=True)
os.makedirs('../homework/models/bert-base-chinese', exist_ok=True)

# 加载和预处理数据
try:
    dataset_df = pd.read_csv('作业数据-waimai_10k.csv', sep=',', header=None)
    print(f"成功加载数据，共{len(dataset_df)}条记录")
except FileNotFoundError:
    print("数据文件未找到，请检查文件路径")
    exit()

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 提取前1000个样本
sample_size = min(1000, len(dataset_df) - 1)  # 确保不超过数据范围
labels = lbl.fit_transform(dataset_df[0].values[1:sample_size + 1])
texts = list(dataset_df[1].values[1:sample_size + 1])

print(f"标签类别: {lbl.classes_}")
print(f"样本分布: {np.bincount(labels)}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,  # 文本数据
    labels,  # 对应的数字标签
    test_size=0.2,  # 测试集比例为20%
    stratify=labels,  # 确保训练集和测试集的标签分布一致
    random_state=42  # 设置随机种子以确保结果可复现
)

print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")

# 从预训练模型加载分词器和模型
try:
    tokenizer = BertTokenizer.from_pretrained('../homework/models/bert-base-chinese', local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        '../homework/models/bert-base-chinese',
        num_labels=len(lbl.classes_),  # 类别数量根据数据自动确定
        problem_type="single_label_classification"  # 明确指定问题类型
    )
    print("成功加载BERT模型和分词器")
except Exception as e:
    print(f"加载模型失败: {e}")
    # 如果本地模型不存在，尝试从HuggingFace下载
    try:
        print("尝试从HuggingFace下载模型...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=len(lbl.classes_),
            problem_type="single_label_classification"
        )
        # 保存模型到本地以备后用
        tokenizer.save_pretrained('../homework/models/bert-base-chinese')
        model.save_pretrained('../homework/models/bert-base-chinese')
        print("模型下载并保存成功")
    except Exception as e2:
        print(f"下载模型也失败: {e2}")
        exit()

# 使用分词器对训练集和测试集的文本进行编码
def tokenize_function(examples):
    return tokenizer(examples, truncation=True, padding=True, max_length=64)

train_encodings = tokenize_function(x_train)
test_encodings = tokenize_function(x_test)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],  # 文本的token ID
    'attention_mask': train_encodings['attention_mask'],  # 注意力掩码
    'labels': train_labels.astype(np.int64)  # 确保标签是int64类型
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels.astype(np.int64)  # 确保标签是int64类型
})

# 自定义数据整理函数
def custom_collate_fn(features):
    batch = {
        'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
        'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
        'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)  # 确保是long类型
    }
    return batch

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='../homework/bert/',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  # 新版本
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# 实例化 Trainer
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,  # 用于计算评估指标的函数
    tokenizer=tokenizer,  # 添加分词器以便保存
    data_collator=custom_collate_fn,  # 添加自定义数据整理器
)

# 开始训练模型
print("开始训练模型...")
train_result = trainer.train()

# 在测试集上进行最终评估
eval_result = trainer.evaluate()
print(f"最终评估结果: {eval_result}")

# 保存最佳模型
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    print(f"最佳模型保存在: {best_model_path}")

    # 保存完整的模型和分词器，方便后续使用
    trainer.save_model('../homework/bert/best_model')
    tokenizer.save_pretrained('../homework/bert/best_model')

    # 保存标签编码器
    joblib.dump(lbl, '../homework/bert/label_encoder.pkl')

    print("模型、分词器和标签编码器已保存")
else:
    print("未找到最佳模型检查点，保存最终模型")
    trainer.save_model('../homework/bert/final_model')
    tokenizer.save_pretrained('../homework/bert/final_model')
    joblib.dump(lbl, '../homework/bert/label_encoder.pkl')

print("训练完成!")

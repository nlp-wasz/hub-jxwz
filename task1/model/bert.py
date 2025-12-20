import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset


def model_for_bert(
        data_path="D:/nlp20/week4/task1/assets/dataset/waimai_10k.csv",
        model_path="E:/models/google-bert/bert-base-chines",
        num_train_epochs=6,
        per_device_batch_size=8,
        max_seq_length=64,
        test_size=0.2
):
    """
    加载外卖评价数据集，训练BERT文本分类模型，并返回核心工具（分词器、训练后模型、标签编码器）
    """
    # -------------------------- 1. 数据加载与预处理 --------------------------
    print("=" * 50)
    print("开始加载并预处理数据集...")
    try:
        dataset_df = pd.read_csv(data_path, sep=",")
        print(f"成功加载数据集，总行数：{len(dataset_df)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据集文件未找到，请检查路径：{data_path}")

    # 提取文本和标签（前100条样本）
    texts = list(dataset_df["review"].values[:100])
    raw_labels = dataset_df["label"].values[:100]

    # 标签编码
    label_encoder = LabelEncoder()
    num_labels = label_encoder.fit_transform(raw_labels)

    # 分割训练集与测试集
    x_train, x_test, train_labels, test_labels = train_test_split(
        texts,
        num_labels,
        test_size=test_size,
        stratify=num_labels,
        random_state=42
    )

    # 打印数据分布信息
    num_actual_labels = len(label_encoder.classes_)
    print(f"\n数据集实际类别数：{num_actual_labels}")
    print(f"类别映射关系：{dict(zip(range(num_actual_labels), label_encoder.classes_))}")
    print(f"训练集样本数：{len(x_train)}，标签分布：{np.bincount(train_labels)}")
    print(f"测试集样本数：{len(x_test)}，标签分布：{np.bincount(test_labels)}")
    print("数据集预处理完成！")
    print("=" * 50)

    # -------------------------- 2. 加载BERT分词器与预训练模型 --------------------------
    print("\n开始加载BERT分词器与预训练模型...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_actual_labels,
        ignore_mismatched_sizes=True
    )
    print(f"成功加载预训练模型：{model_path}")
    print(f"模型输出类别数：{model.num_labels}（与数据集类别数一致）")
    print("=" * 50)

    # -------------------------- 3. 文本编码（核心修改：调整标签数据类型） --------------------------
    print("\n开始对文本数据进行编码（Tokenize）...")
    # 训练集编码
    train_encodings = tokenizer(
        x_train,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt"
    )
    # 测试集编码
    test_encodings = tokenizer(
        x_test,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # 转换为Hugging Face Dataset格式
    # 核心修复：单类别时使用torch.float32类型，多类别时使用torch.long
    label_dtype = torch.float32 if num_actual_labels == 1 else torch.long

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": torch.tensor(train_labels, dtype=label_dtype)  # 修复数据类型
    })
    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": torch.tensor(test_labels, dtype=label_dtype)  # 修复数据类型
    })
    print(f"训练集编码完成，样本格式：{train_dataset[0].keys()}")
    print(f"标签数据类型：{label_dtype}（自动适配类别数）")
    print(f"测试集编码完成，样本数量：{len(test_dataset)}")
    print("=" * 50)

    # -------------------------- 4. 定义评估指标（准确率） --------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # 单类别时需要特殊处理（sigmoid激活）
        if num_actual_labels == 1:
            predictions = (logits > 0.5).astype(int).flatten()  # 大于0.5视为正例
        else:
            predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": round(accuracy, 4)}

    # -------------------------- 5. 配置训练参数 --------------------------
    print("\n配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        fp16=torch.cuda.is_available(),
        disable_tqdm=False
    )
    print("训练参数配置完成，即将开始训练！")
    print("=" * 50)

    # -------------------------- 6. 初始化Trainer并开始训练 --------------------------
    print("\n开始训练模型...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 启动训练
    trainer.train()

    # -------------------------- 7. 最终评估与模型返回 --------------------------
    print("\n训练完成！开始最终测试集评估...")
    final_eval_results = trainer.evaluate()
    print(f"\n最终评估结果：")
    print(f"测试集损失（eval_loss）：{round(final_eval_results['eval_loss'], 4)}")
    print(f"测试集准确率（eval_accuracy）：{round(final_eval_results['eval_accuracy'], 4)}")
    print("=" * 50)

    return tokenizer, model, label_encoder


if __name__ == "__main__":
    try:
        tokenizer, trained_model, label_encoder = model_for_bert()
        print("\n✅ 模型训练全流程完成！")
        print(f"🔧 可用于推理的工具：")
        print(f"   - 分词器：{type(tokenizer).__name__}")
        print(f"   - 训练后模型：{type(trained_model).__name__}")
        print(f"   - 标签编码器：类别映射 {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}")
    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
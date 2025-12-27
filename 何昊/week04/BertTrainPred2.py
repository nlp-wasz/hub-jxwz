import torch
import pandas as pd
from DataType import Request, Response
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# 使用 Hugging Face 中的 transformers 库中的 Trainer，TrainingArguments 模块，自动训练评估
def auto_bert_train_eval(request: Request):
    epochs = request.epochs
    data_size = request.data_size
    half = data_size // 2

    # 1.读取数据
    data = pd.read_csv("./data/作业数据-waimai_10k.csv", sep=",")
    data = pd.concat((data[:half], data[-1:-half - 1:-1]))
    texts = data["review"].values
    labels = data["label"].values

    # 2.划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(list(texts), (labels), test_size=0.2,
                                                                          stratify=labels)

    # 3.AutoTokenizer 自动分词器处理
    tokenizer = AutoTokenizer.from_pretrained("../../../models/google-bert/bert-base-chinese")
    train_encoding = tokenizer(train_texts, max_length=64, padding=True, truncation=True, return_tensors="pt")
    test_encoding = tokenizer(test_texts, max_length=64, padding=True, truncation=True, return_tensors="pt")

    # 4.Dataset数据对象（Hugging Face 中的 datasets 库中的 Dataset 模块）
    train_dataset = Dataset.from_dict({
        "input_ids": train_encoding["input_ids"],
        "attention_mask": train_encoding["attention_mask"],
        "labels": train_labels,
    })
    test_dataset = Dataset.from_dict({
        "input_ids": test_encoding["input_ids"],
        "attention_mask": test_encoding["attention_mask"],
        "labels": test_labels,
    })

    # 5.加载Bert模型对象
    bert_model = AutoModelForSequenceClassification.from_pretrained("../../../models/google-bert/bert-base-chinese",
                                                                    num_labels=2)

    # 6.配置训练参数
    training_arguments = TrainingArguments(
        num_train_epochs=epochs,  # 训练轮数
        per_device_train_batch_size=16,  # 每设备 训练批次样本数
        per_device_eval_batch_size=16,  # 每设备 评估批次样本数
        learning_rate=2e-5,  # 学习率
        weight_decay=0.01,  # 权重降低率
        warmup_steps=500,  # 学习率预热步数（提前加载 500个批次的学习率，learning_rate参数也可以不加）

        output_dir="./data/output",  # 模型及状态 保存目录
        logging_dir="./data/log",  # 日志记录目录
        logging_steps=10,  # 每 训练10个批次，记录一次日志信息
        evaluation_strategy="epoch",  # 每循环一次，评估一次模型
        save_strategy="epoch",  # 每循环一次，保存一次模型信息

        load_best_model_at_end=True,  # 训练结束后，加载效果最好的模型信息
        metric_for_best_model="eval_loss",  # 模型优劣，判断标准
        greater_is_better=False  # True：判断标准越大越好，False：判断标准越小越好
    )

    # 7.模型训练对象
    def compute_func(outputs):
        logits, label = outputs
        # 精度计算方法（自定义 或者 使用sklearn）
        eval_output = logits.argmax(axis=-1)

        return {"accuracy_score": accuracy_score(label, eval_output)}

    train = Trainer(
        model=bert_model,  # 要训练的模型
        args=training_arguments,  # 训练参数
        train_dataset=train_dataset,  # 训练集
        eval_dataset=test_dataset,  # 评估集
        compute_metrics=compute_func  # 评估精度计算方法
    )

    # 8.训练 评估
    train.train()
    train.evaluate()

    # 9.保存模型
    train.save_model("./data/autoTrain")


request = Request(epochs=3, data_size=200)
auto_bert_train_eval(request)

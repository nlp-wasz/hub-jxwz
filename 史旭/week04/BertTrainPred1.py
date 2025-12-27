from DataType import Request, Response
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib


# Bert模型 模型训练 加 预测
def self_bert_train_pred(request: Request) -> Response:
    # 用来记录 每次循环 每批次 损失值
    epoch_batch_loss = []
    # 用来记录 每次循环 每批次 预测精度
    epoch_batch_pred = []

    epochs = request.epochs
    data_size = request.data_size
    half = data_size // 2

    # 1.读取数据
    data = pd.read_csv('./data/作业数据-waimai_10k.csv', sep=",")
    # 获取 标签和文本
    data = pd.concat((data[:half], data[-1:-half - 1:-1]), ignore_index=True)
    texts = data["review"].values
    labels = data["label"].values

    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(list(texts), list(labels), test_size=0.2,
                                                                          stratify=labels)
    # 文本转换为特征向量（transformers.AutoTokenizer）
    bert_tokenizer = AutoTokenizer.from_pretrained('../../../models/google-bert/bert-base-chinese')
    train_vectors = bert_tokenizer(train_texts, max_length=64, padding=True, truncation=True)
    test_vectors = bert_tokenizer(test_texts, max_length=64, padding=True, truncation=True)

    # 2.数据集 对象
    class BertDataset(Dataset):
        def __init__(self, vectors, labels):
            super(BertDataset, self).__init__()
            self.vectors = vectors
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # train_vector 和 test_vector 都是字典，包含input_ids, attention_mask, token_type_ids
            # 每个key都对应着list，将其装换为tensor
            inputs = {key: torch.tensor(value[idx]) for key, value in self.vectors.items()}

            # 添加 标签labels
            inputs["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

            return inputs

    # 3.对象初始化
    # 数据获取对象 Dataset
    train_dataset = BertDataset(train_vectors, train_labels)
    test_dataset = BertDataset(test_vectors, test_labels)

    # 数据加载对象 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # bert模型
    bert_model = AutoModelForSequenceClassification.from_pretrained('../../../models/google-bert/bert-base-chinese',
                                                                    num_labels=2)
    # 优化器
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model.to(device)

    # 4.模型训练
    def bert_train():
        bert_model.train()

        # 记录所有批次 的损失值
        batch_loss = []

        # 批次训练
        for batch, inputs in enumerate(train_dataloader):
            # 获取 训练需要的数据
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 模型训练，计算损失值（outputs：包含 loss 和 logits 两部分数据）
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 反向传播，计算梯度
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1)
            # 参数调整
            optimizer.step()

            print(f"第{epoch + 1}循环，第{batch + 1}批次，模型训练loss：{loss.item()}")

            # 记录所有批次 的损失值
            batch_loss.append(loss.item())

        # 用来记录 每次循环 每批次 损失值
        epoch_batch_loss.append(batch_loss)
        print("-" * 50)

    # 5.模型预测
    def bert_test():
        bert_model.eval()
        total_pred = 0
        total_all = 0

        # 批次预测
        for batch, inputs in enumerate(test_dataloader):
            # 获取 训练需要的数据
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)

            # 禁用梯度计算等操作
            with torch.no_grad():
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

                # 获取输出结果
                logits = outputs.logits

                # 计算精度
                pred_count = torch.sum(logits.argmax(dim=-1) == labels).item()

                print(
                    f"第{epoch + 1}循环，第{batch + 1}批次，模型预测精度：{pred_count}/{len(labels)} = {pred_count / len(labels):.4f}")

                # 整体精度
                total_pred += pred_count
                total_all += len(labels)
        print(
            f"第{epoch + 1}循环，模型整体预测精度：{total_pred}/{total_all} = {total_pred / total_all:.4f}")
        print("=" * 50)

        # 用来记录 每次循环 的所有批次的整体预测精度
        epoch_pred = f"第{epoch + 1}循环，模型整体预测精度：{total_pred}/{total_all} = {total_pred / total_all:.4f}"
        epoch_batch_pred.append(epoch_pred)

    for epoch in range(epochs):
        bert_train()
        bert_test()

    # 模型训练完，保存下来（用于 预测分类）
    joblib.dump(bert_model, "./data/bert_train_model.bin")

    # 构建 响应体
    response = Response(train_loss=epoch_batch_loss, test_precision=epoch_batch_pred, run_time="")

    return response

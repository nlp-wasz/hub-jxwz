import os
import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

# 自定义数据集类，继承自PyTorch的Dataset
# 用于处理编码后的数据和标签，方便后续批量读取
class NewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 获取单个样本的方法
    def __getitem__(self, idx):
        # 从编码字典中提取input_ids, token_types_ids,attention_mask，并转换为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 添加标签，并转换为张量
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    # 返回数据集总样本数的方法
    def __len__(self):
        return len(self.labels)

# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#保存训练好的模型
def save_model(model, tokenizer, lbl, epoch, base_dir="./app/models"):
    """保存模型和相关文件"""
    output_dir = f"{base_dir}/bert-finetuned-epoch{epoch}"
    os.makedirs(output_dir, exist_ok=True)
        
    # 保存模型和分词器
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
        
    # 保存标签编码器
    joblib.dump(lbl, os.path.join(output_dir, "label_encoder.pkl"))
        
    print(f"Model saved to {output_dir}")
    return output_dir

# -------------------------- 主训练函数 --------------------------
def main_train():
    # -------------------------- 1. 数据准备 --------------------------
    # 加载数据集，指定分隔符为制表符，第一行为列名
    dataset = pd.read_csv("./assets/waimai_10k.csv", sep=",")
    dataset.columns = dataset.columns.str.strip()  # 去除列名两端的空格

    print("Label数据类型:", type(dataset['label'])) #pandas.core.series.Series
    print("Label数据类型:", type(dataset['label'].values))    #numpy.ndarray
    print("review数据类型:", type(dataset['review'].iloc[:5]))  #pandas.core.series.Series
    print(dataset.head(5))
    
    dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    subset = dataset_shuffled.iloc[:500]

    # 初始化并拟合标签编码器，将文本标签转换为数字标签（如0, 1, 2...）
    lbl = LabelEncoder()
    labels = lbl.fit_transform(subset['label']) #Transform labels to normalized encoding
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # 将数据按8:2的比例分割为训练集和测试集
    # stratify 参数确保训练集和测试集中各类别的样本比例与原始数据集保持一致
    x_train, x_test, train_label, test_label = train_test_split(
        list(subset['review']),
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 加载BERT预训练的分词器（Tokenizer）
    # 分词器负责将文本转换为模型可识别的输入ID、注意力掩码等
    tokenizer = BertTokenizer.from_pretrained('./assets/models/google-bert/bert-base-chinese')
    # 打印 tokenizer 基本信息
    print("=== Tokenizer 基本信息 ===")
    print(f"Tokenizer 类型: {type(tokenizer)}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"模型最大长度: {tokenizer.model_max_length}")

    # 对训练集和测试集的文本进行编码
    # truncation=True：如果句子长度超过max_length，则截断
    # padding=True：将所有句子填充到max_length
    # max_length=64：最大序列长度
    train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    print("所有键:", list(train_encoding.keys()))

    for key, value in train_encoding.items():
        print(f"=== {key} ===")
        print(f"数据类型: {type(value)}")
        print(f"数据形状: {len(value)} × {len(value[0])}")
        print(f"示例数据: {value[0]}")
        print()

    test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

    # -------------------------- 2. 数据集和数据加载器 --------------------------
    # 实例化自定义数据集
    train_dataset = NewDataset(train_encoding, train_label) # 单个样本读取的数据集
    test_dataset = NewDataset(test_encoding, test_label)

    #import pdb; pdb.set_trace()

    # 使用DataLoader创建批量数据加载器
    # batch_size=16：每个批次包含16个样本
    # shuffle=True：在每个epoch开始时打乱数据，以提高模型泛化能力
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # 批量读取样本
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    for i, batch_data in enumerate(train_loader, 1):  # 1表示从1开始计数
        print(f"batch_data[{i}]所有键:", list(batch_data.keys()))
        print(f"batch{i} len= {len(batch_data)}")
        if i>=2:
            break
    # -------------------------- 3. 模型和优化器 --------------------------
    # 加载BERT用于序列分类的预训练模型
    # num_labels=12：指定分类任务的类别数量
    # https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/bert#transformers.BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('./assets/models/google-bert/bert-base-chinese', num_labels=17)

    # 设置设备，优先使用CUDA（GPU），否则使用CPU
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 将模型移动到指定的设备上
    model.to(device)    #type: ignore

    # 定义优化器，使用AdamW，lr是学习率
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # -------------------------- 4.训练和验证函数 --------------------------
    # 定义训练函数
    def train():
        # 设置模型为训练模式
        model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(train_loader)

        # 遍历训练数据加载器
        for batch in train_loader:
            # 清除上一轮的梯度
            optim.zero_grad()

            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 执行前向传播，得到损失和logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 自动计算损失
            loss = outputs[0]
            total_train_loss += loss.item()

            # 反向传播计算梯度
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新模型参数
            optim.step()

            iter_num += 1
            # 每100步打印一次训练进度
            if (iter_num % 100 == 0):
                print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                    epoch, iter_num, loss.item(), iter_num / total_iter * 100))

        # 打印平均训练损失
        print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


    # 定义验证函数
    def validation():
        # 设置模型为评估模式
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        # 遍历测试数据加载器
        for batch in test_dataloader:
            # 在验证阶段，不计算梯度
            with torch.no_grad():
                # 将批次数据移动到指定设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 执行前向传播
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()
            # 将logits和标签从GPU移动到CPU，并转换为numpy数组
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 计算平均准确率
        avg_loss=total_eval_loss/len(test_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        print("Accuracy: %.4f" % (avg_val_accuracy))
        print("Average testing loss: %.4f" % (avg_loss))
        print("-------------------------------")
        return avg_loss, avg_val_accuracy
    
    # -------------------------- 5. 主训练循环 --------------------------
    # 循环训练4个epoch
    best_accuracy=0
    for epoch in range(4):
        print("------------Epoch: %d ----------------" % epoch)
        # 训练模型
        train()
        # 验证模型
        val_loss, val_accuracy=validation()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, tokenizer, lbl, epoch)
            print(f"🎉 新的最佳模型保存，准确率: {val_accuracy:.4f}")
    print(f"\n训练完成！最佳准确率: {best_accuracy:.4f}")


# 如果是直接运行这个文件，则执行训练
if __name__ == "__main__":
    main_train()
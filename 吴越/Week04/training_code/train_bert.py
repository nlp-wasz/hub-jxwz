import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import joblib



dataset_df=pd.read_csv('../assets/dataset/waimai_10k.csv',sep=',')


#初始化标签编码器，将文本标签转换为数字标签
lbl=LabelEncoder()
#将数据集中标签数据转成数字标签
labels=lbl.fit_transform(dataset_df['label'].values)
#提取文本内容
texts=list(dataset_df['review'])

#分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)

#加载分词器
tokenizer=BertTokenizer.from_pretrained("../assets/models/google-bert/bert-base-chinese")
model=BertForSequenceClassification.from_pretrained("../assets/models/google-bert/bert-base-chinese",num_labels=2)

train_encodings=tokenizer(x_train,truncation=True,padding=True,max_length=64)
test_encodings=tokenizer(x_test,truncation=True,padding=True,max_length=64)

#自定义数据集
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self, idx):
        item={key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels']=torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

#实例化自定义的数据集
train_dataset=NewsDataset(train_encodings,train_labels)
test_dataset=NewsDataset(test_encodings,test_labels)

#定义数据加载器
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=16,shuffle=True)


# 设置设备，优先使用CUDA（GPU），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定的设备上
model.to(device)


# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#定义训练函数
def train(epoch):
    model.train()
    total_train_loss=0
    iter_num=0
    total_iter=len(train_loader)

    for batch in train_loader:
        optim.zero_grad()

        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        outputs=model(input_ids,attention_mask=attention_mask,labels=labels)
        loss=outputs[0]
        total_train_loss+=loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)

        optim.step()

        iter_num+=1

        # 每100步打印一次训练进度
        if (iter_num % 100 == 0):
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
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
    for batch in test_loader:
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
    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
    print("-------------------------------")

    return avg_val_accuracy


#保存训练好的模型
def save_model(model, tokenizer, lbl, epoch, base_dir="../assets/weights"):
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


def main_train():
    best_accuracy = 0
    for epoch in range(4):
        print("------------Epoch: %d ----------------" % epoch)
        # 训练模型
        train(epoch)
        # 验证模型
        val_accuracy = validation()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, tokenizer, lbl, epoch)
            print(f"新的最佳模型保存，准确率: {val_accuracy:.4f}")
    print(f"\n训练完成！最佳准确率: {best_accuracy:.4f}")


# 如果是直接运行这个文件，则执行训练
if __name__ == "__main__":
    main_train()
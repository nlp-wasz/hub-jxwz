import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

dataset=pd.read_csv("./dataset.csv",sep='\t',header=None)
texts=dataset[0].tolist()
string_labels = dataset[1].tolist()
#set(string_labels)：将字符串标签列表去重，得到标签集合
#enumerate(set(string_labels)): 遍历这个唯一标签集合，同时获取索引和标签值
#{label: i for i, label in ...}: 创建字典，键是标签，值是对应的索引,例如：{'Alarm-Update': 0, 'Video-Play': 1}
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
#将字符串标签转换成字典中对应的数值，并且创建新列表
numerical_labels = [label_to_index[label] for label in string_labels]
#创建字符到索引的映射字典
char_to_index = {'<pad>': 0}
#遍历所有文本的每个字符，为每个新字符分配一个唯一的递增索引，索引从1开始
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)



index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 动态模型类
class DynamicNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(DynamicNet, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size=hidden_size
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_and_record_loss(model,dataloader,num_epochs):
    #定义损失函数
    criterion=nn.CrossEntropyLoss()
    #定义优化器
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    loss_list=[]
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for inputs,labels in dataloader:
            optimizer.zero_grad()#梯度清空
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()#损失函数反向传播
            optimizer.step()#参数更新
            running_loss+=loss.item()
        avg_loss=running_loss/len(dataloader)
        loss_list.append(avg_loss)
    return loss_list

hidden_layers_config=[
    [16],[32],[64],
    [16,16],[32,32],[64,64],
    [16,16,16],[32,32,32],[64,64,64]
]

output_dim = len(label_to_index)
input_dim = len(char_to_index)
num_epochs=10

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

loss_curves=[]
labels=[]
for config in hidden_layers_config:
    print(f"Training struct:{config} ")
    model=DynamicNet(input_dim,output_dim,config)
    loss_history=train_and_record_loss(model,dataloader,num_epochs)
    loss_curves.append(loss_history)
    labels.append(f"{len(config)}layers,{config[0]} per layer")

plt.figure(figsize=(12,7))
for loss,label in zip(loss_curves,labels):
    plt.plot(range(1,num_epochs+1),loss,label=label)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison of Different Network Structures")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




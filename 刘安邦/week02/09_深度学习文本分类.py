"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，
对比模型的loss变化。
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
# dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
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

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        if len(hidden_dims) > 1:
            for i in range(len(hidden_dims) - 1):
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.__total_para = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # 手动实现每层的计算
        for layer in self.layers:
            x = layer(x)
        return x

    def total_para(self):
        return self.__total_para

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)
num_epochs = 50

loss_records = list()
hidden_dims_list = [[128],[64,64],[43,43,43],[32,32,32,32]]
# hidden_dims_list = [[128],[64],[32]]
# hidden_dims_list = [[64],[64,64],[64,64,64]]

for i in range(len(hidden_dims_list)):
    print('__', hidden_dims_list[i], '__')
    print('input_dim: ', vocab_size)
    print('output_dim: ', output_dim)
    hidden_dims = hidden_dims_list[i]
    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)  # 维度和精度有什么关系？

    print('total_para: ', model.total_para())

    criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_record = list()

    for epoch in range(num_epochs):  # 每个epoch都用12000个样本， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 100 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        loss_record.append(running_loss / len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    loss_records.append(loss_record)

# 输出loss曲线对比
plt.figure(figsize=(10, 6))
for i in range(len(loss_records)):
    # plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
    plt.plot(list(range(num_epochs)), loss_records[i], label=f'Model {i}', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()

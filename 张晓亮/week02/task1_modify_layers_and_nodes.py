from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv(r"dataset.csv", sep="\t", header=None)
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
    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int, activation: str="relu"):   # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.model = self.build_model()

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()

    def build_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        layers.append(self.get_activation())
        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))
            layers.append(self.get_activation())
        layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


def train(model_new):
    num_epochs = 10
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model_new.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model_new(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 50 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return model_new


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据
index_to_label = {i: label for label, i in label_to_index.items()}
output_dim = len(label_to_index)
criterion = nn.CrossEntropyLoss()

for i, hidden_dim in enumerate([[512, 256, 128, 64, 32], [512, 256, 64, 32], [512, 128, 32], [512, 128], [512, 64], [512, 32], [128, 64], [64, 32], [128, 32]]):
    """
        [512, 256, 128, 64] 4个隐藏层，节点分别是512， 256， 128， 64
        [512, 64] 2个隐藏层，节点分别是512，64
        [128, 64] 2个隐藏层，节点分别是128，64
        [512, 128] 2个隐藏层，节点分别是128，64
    """
    print("-" * 20 + f"第{i+1}个模型训练开始：{hidden_dim}" + "-" * 20)
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    trained_model = train(model)

    new_text = "帮我导航到北京"
    predicted_class = classify_text(new_text, trained_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"第{i+1}个模型训练输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text(new_text_2, trained_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"第{i+1}个模型训练输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


# 层数越多需要训练的epoch越大？
# 隐藏层相邻节点数差距越大训练效果越明显，loss收敛越快？
# 节点数越大训练越慢？


"""
--------------------第1个模型训练开始：[512, 256, 128, 64, 32]--------------------
Epoch [1/10], Loss: 2.4257
Epoch [2/10], Loss: 2.3727
Epoch [3/10], Loss: 2.3564
Epoch [4/10], Loss: 2.3531
Epoch [5/10], Loss: 2.3512
Epoch [6/10], Loss: 2.3507
Epoch [7/10], Loss: 2.3515
Epoch [8/10], Loss: 2.3512
Epoch [9/10], Loss: 2.3497
Epoch [10/10], Loss: 2.3490
第1个模型训练输入 '帮我导航到北京' 预测为: 'Video-Play'
第1个模型训练输入 '查询明天北京的天气' 预测为: 'Video-Play'
--------------------第2个模型训练开始：[512, 256, 64, 32]--------------------
Epoch [1/10], Loss: 2.4285
Epoch [2/10], Loss: 2.3735
Epoch [3/10], Loss: 2.3555
Epoch [4/10], Loss: 2.3497
Epoch [5/10], Loss: 2.3469
Epoch [6/10], Loss: 2.3431
Epoch [7/10], Loss: 2.3330
Epoch [8/10], Loss: 2.2748
Epoch [9/10], Loss: 1.9884
Epoch [10/10], Loss: 1.5292
第2个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第2个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第3个模型训练开始：[512, 128, 32]--------------------
Epoch [1/10], Loss: 2.4376
Epoch [2/10], Loss: 2.3803
Epoch [3/10], Loss: 2.3525
Epoch [4/10], Loss: 2.3240
Epoch [5/10], Loss: 2.2023
Epoch [6/10], Loss: 1.6135
Epoch [7/10], Loss: 0.9900
Epoch [8/10], Loss: 0.6401
Epoch [9/10], Loss: 0.4978
Epoch [10/10], Loss: 0.4187
第3个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第3个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第4个模型训练开始：[512, 128]--------------------
Epoch [1/10], Loss: 2.4236
Epoch [2/10], Loss: 2.2759
Epoch [3/10], Loss: 1.8221
Epoch [4/10], Loss: 1.1077
Epoch [5/10], Loss: 0.6997
Epoch [6/10], Loss: 0.5238
Epoch [7/10], Loss: 0.4364
Epoch [8/10], Loss: 0.3811
Epoch [9/10], Loss: 0.3414
Epoch [10/10], Loss: 0.3106
第4个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第4个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第5个模型训练开始：[512, 64]--------------------
Epoch [1/10], Loss: 2.4147
Epoch [2/10], Loss: 2.2660
Epoch [3/10], Loss: 1.7506
Epoch [4/10], Loss: 1.0711
Epoch [5/10], Loss: 0.6912
Epoch [6/10], Loss: 0.5269
Epoch [7/10], Loss: 0.4444
Epoch [8/10], Loss: 0.3939
Epoch [9/10], Loss: 0.3521
Epoch [10/10], Loss: 0.3206
第5个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第5个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第6个模型训练开始：[512, 32]--------------------
Epoch [1/10], Loss: 2.4241
Epoch [2/10], Loss: 2.2367
Epoch [3/10], Loss: 1.7558
Epoch [4/10], Loss: 1.1054
Epoch [5/10], Loss: 0.7073
Epoch [6/10], Loss: 0.5388
Epoch [7/10], Loss: 0.4533
Epoch [8/10], Loss: 0.3956
Epoch [9/10], Loss: 0.3568
Epoch [10/10], Loss: 0.3282
第6个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第6个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第7个模型训练开始：[128, 64]--------------------
Epoch [1/10], Loss: 2.4249
Epoch [2/10], Loss: 2.3140
Epoch [3/10], Loss: 1.9941
Epoch [4/10], Loss: 1.2913
Epoch [5/10], Loss: 0.7960
Epoch [6/10], Loss: 0.5682
Epoch [7/10], Loss: 0.4642
Epoch [8/10], Loss: 0.4069
Epoch [9/10], Loss: 0.3635
Epoch [10/10], Loss: 0.3309
第7个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第7个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第8个模型训练开始：[64, 32]--------------------
Epoch [1/10], Loss: 2.4056
Epoch [2/10], Loss: 2.2326
Epoch [3/10], Loss: 1.6862
Epoch [4/10], Loss: 1.0444
Epoch [5/10], Loss: 0.6747
Epoch [6/10], Loss: 0.5201
Epoch [7/10], Loss: 0.4437
Epoch [8/10], Loss: 0.3952
Epoch [9/10], Loss: 0.3590
Epoch [10/10], Loss: 0.3289
第8个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第8个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
--------------------第9个模型训练开始：[128, 32]--------------------
Epoch [1/10], Loss: 2.4316
Epoch [2/10], Loss: 2.2908
Epoch [3/10], Loss: 1.8821
Epoch [4/10], Loss: 1.3101
Epoch [5/10], Loss: 0.8710
Epoch [6/10], Loss: 0.6321
Epoch [7/10], Loss: 0.5101
Epoch [8/10], Loss: 0.4379
Epoch [9/10], Loss: 0.3920
Epoch [10/10], Loss: 0.3547
第9个模型训练输入 '帮我导航到北京' 预测为: 'Travel-Query'
第9个模型训练输入 '查询明天北京的天气' 预测为: 'Weather-Query'
"""

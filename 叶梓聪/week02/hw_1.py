import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 数据加载和预处理
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
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


# 动态可配置的模型类
class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, layer_config, output_dim):
        """
        input_dim: 输入维度 (词汇表大小)
        layer_config: 层配置列表，每个元素是该层的神经元数
                    例如 [128] 表示单隐藏层128个神经元
                    [256, 128] 表示两个隐藏层，256→128
        output_dim: 输出层维度 (类别数量)
        """
        super(ConfigurableClassifier, self).__init__()

        self.layers = nn.ModuleList()
        layer_sizes = [input_dim] + layer_config + [output_dim]

        # 创建所有隐藏层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # 输出层前不加ReLU
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.2))  # 减少过拟合

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 训练函数，用于不同模型配置
def train_model(layer_config, hidden_dims_desc, dataloader, vocab_size, output_dim):
    model = ConfigurableClassifier(vocab_size, layer_config, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"配置 {hidden_dims_desc} - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model, np.array(epoch_losses)


# 不同模型配置实验
output_dim = len(label_to_index)
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义要测试的不同配置
configurations = [
    {"layer_config": [], "desc": "0层（线性分类器）"},  # 输入直接到输出
    {"layer_config": [64], "desc": "1层(64节点)"},  # 单隐藏层
    {"layer_config": [128], "desc": "1层(128节点)"},  # 原始配置
    {"layer_config": [256], "desc": "1层(256节点)"},  # 更宽隐藏层
    {"layer_config": [64, 32], "desc": "2层(64,32)"},  # 两层逐渐缩小
    {"layer_config": [256, 128], "desc": "2层(256,128)"},  # 宽→中等两层
    {"layer_config": [512, 256, 128], "desc": "3层(512,256,128)"}  # 三层深度
]

# 存储结果
results = {}
models = {}

# 训练所有配置
for config in configurations:
    print(f"\n===== 训练配置: {config['desc']} =====")
    model, losses = train_model(
        config["layer_config"],
        config["desc"],
        dataloader,
        vocab_size,
        output_dim
    )
    results[config["desc"]] = losses
    models[config["desc"]] = model


# 比较最终损失
print("\n===== 模型性能总结 =====")
final_losses = []
for desc, losses in results.items():
    final_loss = losses[-1]
    final_losses.append((desc, final_loss))
    print(f"{desc}: 最终损失 = {final_loss:.4f}")

# 找出最佳模型
best_config = min(final_losses, key=lambda x: x[1])
print(f"\n最佳模型: {best_config[0]} (损失 = {best_config[1]:.4f})")
best_model = models[best_config[0]]


# 使用最佳模型进行预测
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


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n输入 '{new_text}' 预测为: '{predicted_class}' (使用最佳模型)")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}' (使用最佳模型)")

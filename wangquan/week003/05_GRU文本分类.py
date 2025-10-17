import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 读取数据集，数据集是每一行第一个元素是一句中文文本，第二个元素是字符串类型的一个类别
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
# 获取数据集中第一列的所有元素，生成一个中文文本的列表
texts = dataset[0].tolist()
# 获取数据集中第二列的所有元素，生成一个字符标签的列表
string_labels = dataset[1].tolist()
# print("texts:", texts)
# print("string_labels", string_labels)

# 把字符标签的列表转化成集合去重，然后分别获取到标签的索引和标签本身，以标签：索引的方式生成到字典里面去，保证每一个标签都有唯一的索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# print("label_to_index:", label_to_index)
# 用上面字典里面的标签和索引的对应关系，来把标签列表里面的标签都转换成索引，生成一个新的列表
numerical_labels = [label_to_index[label] for label in string_labels]
# print("numerical_labels", numerical_labels)

# 初始化一个包含填充符<pad>的字典，索引为0
char_to_index = {'<pad>': 0}
# 遍历之前生成的中文文本列表里面的每一个元素，也就是每一个中文句子
for text in texts:
    # 遍历每一个中文句子中的每一个中文字符
    for char in text:
        # 如果单个的中文字符在字符字典里面
        if char not in char_to_index:
            # 以字符为键，以当前字典长度为值，组成键值对
            char_to_index[char] = len(char_to_index)
# print("char_to_index:", char_to_index)
# print("char_to_index:", char_to_index.items())
# print("char_to_index.items():", type(char_to_index.items()))

index_to_char = {i: char for char, i in char_to_index.items()}
# print("index_to_char:", index_to_char)

vocab_size = len(char_to_index)

max_len = 40

class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        print("self.labels", self.labels)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
lstm_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
print("lstm_dataset", lstm_dataset)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
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
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


# 定义GRU模型
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # GRU层
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         # 全连接层，将GRU的输出映射到输出大小
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # 初始化隐藏状态
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         # 前向传播GRU
#         out, _ = self.gru(x, h0)
#         # 取最后一个时间步的输出
#         out = out[:, -1, :]
#         # 通过全连接层得到最终输出
#         out = self.fc(out)
#         return out
#
#
# # 输入维度
# input_size = 10
# # 隐藏层维度
# hidden_size = 20
# # GRU层数
# num_layers = 2
# # 输出维度
# output_size = 1
#
# # 创建GRU模型实例
# model = GRUModel(input_size, hidden_size, num_layers, output_size)
#
# # 随机生成输入数据
# x = torch.randn(32, 5, input_size)  # 批次大小为32，序列长度为5
#
# # 前向传播
# output = model(x)
#
# print(output.shape)
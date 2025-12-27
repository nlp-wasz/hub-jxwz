# 1、 理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers
# 最终 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py
# 2、阅读项目计划书  + 初步项目代码，写清楚四个模型的优缺点，形成一个word/markdown提交。
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# texts, numerical_labels, char_to_index, max_len
data = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
data[2] = data[1].factorize()[0]

texts = data[0].tolist()
numerical_labels = data[2].tolist()
index_to_label = dict(data[[2, 1]].drop_duplicates().values)
label_to_index = dict(data[[1, 2]].drop_duplicates().values)

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
max_len = 40


# from torch.utils.data import Dataset,DataLoader 文档地址：https://docs.pytorch.ac.cn/docs/stable/data.html
# 继承Dataset必须实现__getitem__和__len__
class CharGruDataset(Dataset):
    def __init__(self, texts, numerical_labels, char_to_index, max_len):
        self.texts = texts
        self.numerical_labels = numerical_labels
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.texts[index]
        text_vec = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]  # 截取长度max_len
        text_vec += [0] * (self.max_len - len(text_vec))  # 补0
        return torch.tensor(text_vec, dtype=torch.long), numerical_labels[index]

    def __len__(self):
        return len(self.texts)


# PyTorch 数据加载
gru_dataset = CharGruDataset(texts, numerical_labels, char_to_index, max_len)
data_loader = DataLoader(gru_dataset, batch_size=32, shuffle=True)


# import torch.nn as nn 文档地址:https://docs.pytorch.ac.cn/docs/stable/nn.html
# forward 所有子类都应重写此方法。
class GruClassifier(nn.Module):
    # num_embeddings (int) – 嵌入词典的大小
    # embedding_dim (int) – 每个嵌入向量的大小
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, output_dim):
        # 必须先调用父类的 __init__()，然后才能对子类进行赋值。
        super(GruClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # input_size – 输入 x 中预期特征的数量
        # hidden_size – 隐藏状态 h 中的特征数量
        # num_layers – 循环层的数量。例如，设置 num_layers=2 意味着将两个 GRU 堆叠在一起形成一个 堆叠 GRU，其中第二个 GRU 接收第一个 GRU 的输出并计算最终结果。默认值：1
        # bias – 如果为 False，则层不使用偏差权重 b_ih 和 b_hh。默认值：True
        # batch_first – 如果为 True，则输入和输出张量以 (batch, seq, feature) 形式提供，而不是 (seq, batch, feature)。请注意，这不适用于隐藏状态或单元状态。有关详细信息，请参阅下面的输入/输出部分。默认值：False
        # dropout – 如果非零，则在除最后一层之外的每个 GRU 层的输出上引入一个 Dropout 层，dropout 概率等于 dropout。默认值：0
        # bidirectional – 如果为 True，则成为双向 GRU。默认值：False
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # 输出：output, h_n
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


model = GruClassifier(len(char_to_index), 64, 128, len(index_to_label))
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for index, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 50 == 0:
            print(f"Batch 个数 {index}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")


def classifier_text_gru(model, text, char_to_index, max_len, index_to_label):
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


new_text = "帮我导航到北京"
predicted_class = classifier_text_gru(model, new_text, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classifier_text_gru(model, new_text_2, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

new_text_3 = "深度拿上来的那卢卡斯你可别放弃我放弃就能看"
predicted_class_3 = classifier_text_gru(model, new_text_3, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_3}' 预测为: '{predicted_class_3}'")
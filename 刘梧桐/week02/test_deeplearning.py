import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 加载文本数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
labels = dataset[1].tolist()

# 标签编码
# 将标签改为{"Travel":1,"Music":2}的字典形式
label_to_index = {label: i for i, label in enumerate(set(labels))}
# 将原来的标签变成数字，方便神经网络识别（神经网络只能处理数值数据，不能识别字符串）
number_label = [label_to_index[label] for label in labels]

# 字符词汇表构建
# 预留特殊标记，填充字符
char_to_index = {"<pad>": 0}
for text in texts:
    for char_str in text:
        if char_str not in char_to_index:
            char_to_index[char_str] = len(char_to_index)

# 反向映射  数字:字符
index_to_char = {i: char for i, char in char_to_index.items()}
vocab_size = len(char_to_index)

# 定义文本最大长度，超过截断，不足补充
max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bow_vectors = self._create_bow_vector() # 预计算所有词袋向量

    def _create_bow_vector(self):
        # 将所有的语句转换成数字存到tokenize_texts中
        tokenize_texts = []
        for text in self.texts:
            # 字符转索引
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 不足补0
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenize_texts.append(tokenized)

        # 创建向量集来存储所有语句的向量
        bow_vectors = []
        for tokenized in tokenize_texts:
            # 创建向量（使用矩阵表示向量）
            bow_vector = torch.zeros(self.vocab_size)
            for index in tokenized:
                if index != 0:
                    # 统计字符出现的次数
                    bow_vector[index] += 1
            # 所有的语句都转成向量存起来
            bow_vectors.append(bow_vector)
        # 堆叠成张量
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]  # 返回样本和标签


# 创建神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dims,dropout_rate = 0.3):
        super(SimpleClassifier, self).__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)


# 创建数据集
char_dataset = CharBoWDataset(texts, number_label, char_to_index, max_len, vocab_size)
# 创建数据集加载器
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 隐藏层的维度
hidden_dims = [512,64]
# 分类数量为输出层的维度
output_dim = len(label_to_index)
# 创建模型
model = SimpleClassifier(input_dim=vocab_size, hidden_dims=hidden_dims)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#####################################################
num_epochs = 10
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()  # 训练模式
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

        train_avg_loss = running_loss / len(dataloader)
        train_losses.append(train_avg_loss)

        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 1. 文本预处理（与训练时相同）
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 2. 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 3. 添加batch维度
    bow_vector = bow_vector.unsqueeze(0)

    # 4. 预测
    model.eval()  # 评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(bow_vector)

    # 5. 获取预测结果
    _, predicted_index = torch.max(output, 1)  # 取最大概率的索引
    predicted_label = index_to_label[predicted_index.item()]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}  # 创建索引到标签的映射

# 测试预测
new_texts =  ["帮我导航到北京", "查询明天北京的天气", "播放周杰伦的音乐"]
for new_text in new_texts:
    predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
    print(predicted_class)

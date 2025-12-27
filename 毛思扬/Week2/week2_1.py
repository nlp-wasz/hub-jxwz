# 1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS系统中常用的中文字体
zh_font = fm.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']


# pandas数据读取
dataset = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
print(type(dataset))
#                                  0                      1
# 0                还有双鸭山到淮阴的汽车票吗13号的           Travel-Query
# 1                          从这里怎么回家           Travel-Query
# 2                 随便播放一首专辑阁楼里的佛里的歌             Music-Play
# 3                        给看一下墓王之王嘛          FilmTele-Play
# 4            我想看挑战两把s686打突变团竞的游戏视频             Video-Play

# 新增一列分类id
dataset[2] = dataset[1].factorize()[0]
#                                  0                      1  2
# 0                还有双鸭山到淮阴的汽车票吗13号的           Travel-Query  0
# 1                          从这里怎么回家           Travel-Query  0
# 过滤重复和重排序
label_id_dataset = dataset[[1, 2]].drop_duplicates().sort_values(2).reset_index(drop=True)
#                         1   2
# 0            Travel-Query   0
# 1              Music-Play   1
# 2           FilmTele-Play   2
# 分类str和分类id的映射
label_to_id = dict(label_id_dataset.values)
# {'Travel-Query': 0, 'Music-Play': 1, 'FilmTele-Play': 2, 'Video-Play': 3, 'Radio-Listen': 4, 'HomeAppliance-Control': 5, 'Weather-Query': 6, 'Alarm-Update': 7, 'Calendar-Query': 8, 'TVProgram-Play': 9, 'Audio-Play': 10, 'Other': 11}
# 分类id和分类str的映射
id_to_label = dict(label_id_dataset[[2, 1]].values)
# {0: 'Travel-Query', 1: 'Music-Play', 2: 'FilmTele-Play', 3: 'Video-Play', 4: 'Radio-Listen', 5: 'HomeAppliance-Control', 6: 'Weather-Query', 7: 'Alarm-Update', 8: 'Calendar-Query', 9: 'TVProgram-Play', 10: 'Audio-Play', 11: 'Other'}

# 构建字符到索引的映射表，<pad>是填充字符，索引为0
char_to_index = {'<pad>': 0}
for text in dataset[0]:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# {'<pad>': 0, '还': 1, '有': 2, '双': 3, '鸭': 4, '山': 5, '到': ...}
vocab_size = len(char_to_index)

index_to_char = dict([(index, char) for char, index in char_to_index.items()])


# {0: '<pad>', 1: '还', 2: '有', 3: '双', 4: '鸭', 5: '山', 6: '到', 7:...}

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        # 初始化参数
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        # 创建词袋向量
        bow_vectors = []
        for text in self.texts:
            bow_vec = torch.zeros(self.vocab_size)
            # 限制文本长度为max_len
            truncated_text = text[:self.max_len]
            for char in truncated_text:
                index = self.char_to_index.get(char, 0)
                if index != 0:  # 排除填充字符（索引为0）
                    bow_vec[index] += 1
            bow_vectors.append(bow_vec)
        return torch.stack(bow_vectors)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

    def __len__(self):
        return len(self.texts)


char_dataset = CharBoWDataset(dataset[0].tolist(), dataset[2].tolist(), char_to_index, 40, vocab_size)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        # 层初始化
        super(SimpleClassifier, self).__init__()

        # 保存网络层列表
        layers = []

        # 添加输入层到第一个隐藏层
        if hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())

            # 添加隐藏层到隐藏层的连接
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.ReLU())

            # 添加最后一个隐藏层到输出层
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            # 如果没有隐藏层，直接连接输入和输出
            layers.append(nn.Linear(input_dim, output_dim))

        # 将列表转换为ModuleList，这样PyTorch可以自动追踪参数
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # 依次通过所有层
        for layer in self.layers:
            x = layer(x)
        return x


data_loader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(id_to_label)

# 定义不同的网络结构进行对比实验
model_configs = [
    {"name": "1层(64节点)", "hidden_dims": [64]},
    {"name": "1层(128节点)", "hidden_dims": [128]},
    {"name": "1层(256节点)", "hidden_dims": [256]},
    {"name": "2层(128-64节点)", "hidden_dims": [128, 64]},
    {"name": "2层(256-128节点)", "hidden_dims": [256, 128]},
    {"name": "3层(256-128-64节点)", "hidden_dims": [256, 128, 64]},
    {"name": "3层(512-256-128节点)", "hidden_dims": [512, 256, 128]},
    {"name": "无隐藏层", "hidden_dims": []}
]


def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    """训练模型并返回每个epoch的loss"""
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses


# 存储所有模型的结果
results = {}

# 对每个配置进行训练和评估
for config in model_configs:
    print(f"\n训练模型: {config['name']}")

    # 创建模型
    model = SimpleClassifier(vocab_size, config['hidden_dims'], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型并记录loss
    losses = train_model(model, data_loader, criterion, optimizer, num_epochs=10)
    results[config['name']] = losses

# 可视化结果
plt.figure(figsize=(12, 8))
for name, losses in results.items():
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同网络结构的Loss变化对比')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果对比
print("\n\n=== 最终Loss对比 ===")
for name, losses in results.items():
    print(f"{name}: 最终Loss = {losses[-1]:.4f}")

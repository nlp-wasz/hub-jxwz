import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 将字符串映射为整数编号，把原始标签转换为数字形式
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符表，将每个字符映射到唯一整数
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 反向索引
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 文本截断长度
max_len = 40

# 文本转向量处理
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    # 将文本转换为向量
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        # 动态构建隐藏层
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        # 将所有层组合为序列
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
# shuffle=True每个epoch从随机数据开始
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

hidden_dim1 = [128]
hidden_dim2 = [128, 64]
hidden_dim3 = [256]
hidden_dim4 = [64]
hidden_dim5 = [256, 128]
output_dim = len(label_to_index)

model1 = SimpleClassifier(vocab_size, hidden_dim1, output_dim)
model2 = SimpleClassifier(vocab_size, hidden_dim2, output_dim)
model3 = SimpleClassifier(vocab_size, hidden_dim3, output_dim)
model4 = SimpleClassifier(vocab_size, hidden_dim4, output_dim)
model5 = SimpleClassifier(vocab_size, hidden_dim5, output_dim)

criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax

optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
optimizer3 = optim.SGD(model3.parameters(), lr=0.01)
optimizer4 = optim.SGD(model4.parameters(), lr=0.01)
optimizer5 = optim.SGD(model5.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

# 存储所有模型和优化器的字典
models = {
    'model1_128': model1,
    'model2_128_64': model2,
    'model3_256': model3,
    'model4_64': model4,
    'model5_256_128': model5
}

optimizers = {
    'model1_128': optimizer1,
    'model2_128_64': optimizer2,
    'model3_256': optimizer3,
    'model4_64': optimizer4,
    'model5_256_128': optimizer5
}

# 训练多个模型
num_epochs = 10
training_losses = {name: [] for name in models.keys()}

print("开始训练多个模型...")
for epoch in range(num_epochs):
    print(f"\n=== 第 {epoch + 1}/{num_epochs} 轮训练 ===")

    # 分别训练每个模型
    for name, model in models.items():
        model.train()
        running_loss = 0.0
        optimizer = optimizers[name]

        # 训练一个epoch
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if idx % 50 == 0:
                print(f"{name} - 批次 {idx}, 损失: {loss.item():.4f}")

        # 计算平均损失
        avg_loss = running_loss / len(dataloader)
        training_losses[name].append(avg_loss)
        print(f"{name} - 本轮平均损失: {avg_loss:.4f}")

print("\n训练完成!")

# 显示最终训练损失
lossmean_list1 = training_losses['model1_128']  # [128] 层
lossmean_list2 = training_losses['model2_128_64']  # [128, 64] 层
lossmean_list3 = training_losses['model3_256']  # [256] 层
lossmean_list4 = training_losses['model4_64']  # [64] 层
lossmean_list5 = training_losses['model5_256_128']  # [64] 层
# 画图展示模型之间的差异（简洁版）
plt.figure(figsize=(12, 6))
plt.plot(lossmean_list1, label="[128]", color="blue")
plt.plot(lossmean_list2, label="[128, 64]", color="red")
plt.plot(lossmean_list3, label="[256]", color="black")
plt.plot(lossmean_list4, label="[64]", color="orange")
plt.plot(lossmean_list5, label="[256_128]", color="brown")
plt.xlabel("Epoch")
plt.ylabel("Loss Mean")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 使用所有模型进行预测
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌",
    "打开空调",
    "设置明天早上7点的闹钟"
]
print("\n=== 各模型预测结果 ===")
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

# 创建index到label的映射
index_to_label = {i: label for label, i in label_to_index.items()}

for text in test_texts:
    print(f"\n输入文本: '{text}'")
    print("-" * 50)

    for name, model in models.items():
        # 设置模型为评估模式
        model.eval()

        # 进行预测
        predicted_class = classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label)
        print(f"{name}: {predicted_class}")

# 可选：选择最佳模型进行进一步使用
# 根据最终损失选择最佳模型
best_model_name = min(training_losses, key=lambda x: training_losses[x][-1])
best_model = models[best_model_name]
print(f"\n最佳模型: {best_model_name} (最终损失: {training_losses[best_model_name][-1]:.4f})")

# 使用最佳模型进行示例预测
print("\n=== 最佳模型预测示例 ===")
example_text = "帮我叫出租车"
prediction = classify_text(example_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入: '{example_text}'")
print(f"预测: '{prediction}'")

# 层数一样的, 节点数多或者少最终曲线几乎重合
# 层数多的反而比层数少的下降慢并且最终loss比层数少的大
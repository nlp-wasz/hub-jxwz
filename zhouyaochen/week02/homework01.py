import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
demo中是一个隐藏层 训练10次 神经元128个
训练三次结果输出:
第一次:
输入 '帮我导航到北京' 
预测为: 'HomeAppliance-Control' 
输入 '明天天气如何' 预测为: 'Music-Play' 
第二次的输出结果是：
输入 '帮我导航到北京' 预测为: 'Alarm-Update' 
输入 '明天天气如何' 预测为: 'Alarm-Update' 
第三次的输出结果是：
输入 '帮我导航到北京' 预测为: 'Weather-Query' 
输入 '明天天气如何' 预测为: 'Weather-Query'

每次训练之后的结果 都不相同
原因分析：
1.模型 SimpleClassifier 里用了 nn.Linear，PyTorch 会默认 随机初始化权重。
2. 训练轮数太少 只训练了 10 个 epoch  对文本分类来说 几乎没学够 模型可能未完全收敛 就被迫停止 所以预测不是很稳定
3. 隐藏层层数不够 表达能力不强
现调整
1.隐藏层调整为3层
2.优化器的选择 选择Adam
3.神经元个数调整为256
4.训练轮数调整为100轮
5.学习率调整
调整后 预测结果明显更为准确:
输入 '帮我导航到北京' 预测为: 'Travel-Query'
输入 '明天天气如何' 预测为: 'Weather-Query'
"""
dataset = pd.read_csv("../Week02/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist() # 数据集第一列 文本
string_labels = dataset[1].tolist() # 数据集第二列 类别

# 类别转换数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 原始的文本构建一个词典，字 -》 数字
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
print(char_to_index)


index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 取文本的前40个字符
max_len = 40

tokenized_texts = []
for text in texts:
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    tokenized_texts.append(tokenized)

labels_tensor = torch.tensor(numerical_labels, dtype=torch.long)

# term frequency
def create_bow_vectors(tokenized_texts, vocab_size):
    bow_vectors = []
    for text_indices in tokenized_texts:
        bow_vector = torch.zeros(vocab_size) # 词典个数长度的向量，存储每个字在这个文本中间出现的次数
        for index in text_indices:
            if index != 0:  # Ignore padding
                bow_vector[index] += 1
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors)

bow_matrix = create_bow_vectors(tokenized_texts, vocab_size)
input_size = vocab_size

# 实际项目中，优先考虑迁移学习，复用已有的网络网络，qwen
"""
隐藏层的层数影响
1.模型的表达能力（复杂度）
神经元个数少：模型的表达能力弱，可能无法捕捉到数据中的复杂模式 → 欠拟合。
神经元个数多：模型的表达能力增强，可以拟合更复杂的函数关系，但也更容易把“噪声”当作模式学习 → 过拟合。
2.训练速度与计算资源
神经元少：参数少，计算快，训练速度快，占用内存小。
神经元多：参数多，计算量大，训练时间变长，对显卡/CPU/内存要求高。
3.模型的泛化能力
合适的神经元数量：能学到有效的规律，在训练集和测试集上都表现良好。
过多神经元：容易过拟合（训练集精度高，但测试集差）。
过少神经元：容易欠拟合（训练集和测试集都表现差）
4.参数量与存储
隐藏层神经元个数直接决定了网络的参数量（权重矩阵的维度）。
例如：
输入层 100 个特征，隐藏层有 50 个神经元 → 参数量约 = 100×50。
如果改成 500 个神经元 → 参数量 = 100×500，多了 10 倍。
"""
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) #nn.Linear Pytorch会随机初始化权重
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

hidden_dim = 256
output_dim = len(label_to_index)
model = SimpleClassifier(input_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss() # 分类损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam 优化器  可以结合梯度 动态调整学习， 0.01 -> 0.001 -> 0.00001

for _ in range(100): #训练的轮数为10 对文本分类来说 几乎没学够 模型可能未完全收敛 就被迫停止 所以预测不是很稳定 调整为100
    model.train()
    optimizer.zero_grad()

    outputs = model(bow_matrix)
    loss = criterion(outputs, labels_tensor)

    loss.backward()
    optimizer.step()

    print(f"Training complete. Loss: {loss.item():.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    # 正向传播，11 神经元的输出
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "明天天气如何"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import jieba
from gensim.models import Word2Vec

# 1.获取数据
data = pd.read_csv("../data/dataset.csv", header=None, sep="\t")

# 获取 文本数据 和 标签
texts = data[0].to_list()
labels = data[1].to_list()

# 转换为 数字特征
# 标签（简单：索引即为数据特征）
label_to_index = {label: index for index, label in enumerate(set(labels))}
index_to_label = {index: label for label, index in label_to_index.items()}
labels_vector = [label_to_index.get(label, 0) for label in labels]
labels_tensor = torch.tensor(labels_vector, dtype=torch.long)

# 文本（构建词汇表，这里采用分词处理，后续训练Word2Vec使用）
char_dict = {"Nan": 0}
for text in texts:
    for char in jieba.cut(text):
        if char not in char_dict:
            char_dict[char] = len(char_dict)


# 2.构建 文本数据对象
class TextDataSet(Dataset):
    def __init__(self, texts, char_dict, labels_tensor, max_sequence):
        super(TextDataSet, self).__init__()
        self.texts = texts
        self.char_dict = char_dict
        self.max_sequence = max_sequence

        self.texts_tesnor = self.get_texts_vector()
        self.labels_tensor = labels_tensor

    def get_texts_vector(self):
        # 转换为 数字特征（词汇表中的索引）
        texts_tensor = []
        for text in self.texts:
            text_vector = [self.char_dict.get(char, 0) for char in list(jieba.cut(text))[:self.max_sequence]]
            text_vector += [0] * (self.max_sequence - len(text_vector))

            # 转换为 tensor
            text_tensor = torch.tensor(text_vector, dtype=torch.long)

            texts_tensor.append(text_tensor)

        return torch.stack(texts_tensor)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (self.texts_tesnor[index], self.labels_tensor[index])


# 3.构建 GRU 神经网络
class GruModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, word2vec, char_dict):
        super(GruModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding 层
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        # 使用 Word2Vec 训练后的特征向量
        # 1.获取创建好的 embedding 层权重数据
        embedding_weights = self.embedding.weight.data
        # 2.循环遍历 char_dict，
        for char, index in char_dict.items():
            # 如果在word2vec中存在与之对应的特征向量，就获取其索引，
            if char in word2vec.wv:
                # 并将embedding层对应索引位置的张量替换为word2vec中的tensor
                embedding_weights[index] = torch.tensor(word2vec.wv[char], dtype=torch.float)

        # GRU
        self.gru = torch.nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # 全链路神经网络（预测输出）
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0):
        embedding = self.embedding(x)
        hidden_all, h_t = self.gru(embedding, h0)

        # 分类 只需要最后时间步 的隐藏状态信息
        # ht：(num_layers, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        h_t = h_t.squeeze(0)

        # out：(batch_size, output_dim)
        out = self.linear(h_t)

        return out

    # GRU 只有隐藏状态，无所初始化细胞状态
    def hidden_init(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


# 4.Word2Vec
# 将 分词后的文本 交给Word2Vec训练，将得到的特征向量传递给 GRU神经网络的 embedding层
text_jieba = [list(jieba.cut(text)) for text in texts]
embedding_dim = 64
word2vec = Word2Vec(
    sentences=text_jieba,  # 训练集
    vector_size=embedding_dim,  # 特征向量维度，这里要传递给后面的 embedding 层
    window=5,  # 要预测值 与 预测值 之间的间隔
    min_count=1,  # 词汇在训练集中 出现的最低频次
    sg=1  # 训练模型 1：
)

# 5.初始化 对象信息
max_sequence = 40
batch_size = 32
vocab_size = len(char_dict)
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 1

# Dataset
dataset = TextDataSet(texts, char_dict, labels_tensor, max_sequence)
# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 自定义 GRU
gru_model = GruModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, word2vec, char_dict)
# 分类损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)

# 6.训练
gru_model.train()
for epoch in range(10):
    for batch_index, (input_tensor, label_tensor) in enumerate(dataloader):
        # h0 初始化
        # batch_size = input_tensor.size(0)
        batch_size = input_tensor.shape[0]
        h0 = gru_model.hidden_init(batch_size)
        # 分离隐藏状态，这里h0每一批次都初始化，不适用上一批次最后时间步隐藏状态（可以不分离）
        # h0 = h0.data

        # 模型训练
        output = gru_model(input_tensor, h0)
        # 计算损失值
        loss = loss_func(output, label_tensor)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度裁剪，防止出现梯度爆炸
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1)
        # 调参
        optimizer.step()

        if batch_index % 50 == 0:
            print(f"第{epoch + 1}次循环，第{batch_index}批次，loss：{loss}")


# 7.预测方法
def pred_func(pred_text, gru_model, char_dict, index_to_label, max_sequence):
    # 转换为 数值特征
    text_vector = [char_dict.get(char, 0) for char in list(jieba.cut(pred_text))[:max_sequence]]
    text_vector += [0] * (max_sequence - len(text_vector))
    text_tensor = torch.tensor(text_vector, dtype=torch.long)

    # text_tensor：(max_sequence, )  ->  (batch_size, max_sequence)
    # 预测单个文本，因此 batch_size = 1
    text_tensor = text_tensor.unsqueeze(0)

    # 文本预测
    gru_model.eval()
    with torch.no_grad():
        # 隐藏状态h0 初始化
        # batch_size = text_tensor.size(0)
        batch_size = text_tensor.shape[0]
        h0 = gru_model.hidden_init(batch_size)

        # 预测结果
        pred_output = gru_model(text_tensor, h0)

        # 获取 最大值 所在索引位置
        pred_maxValue_index = pred_output.argmax(-1)

        # 从 index_to_label 字典中 查询
        pred_result = index_to_label.get(pred_maxValue_index.item(), "无法预测")

    return pred_result


new_text = "帮我导航到北京"
predicted_class = pred_func(new_text, gru_model, char_dict, index_to_label, max_sequence)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = pred_func(new_text_2, gru_model, char_dict, index_to_label, max_sequence)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

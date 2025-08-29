import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 加载数据集
dataset = pd.read_csv("../../../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 创建标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

# 数据集类
class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# GRU 分类器模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # GRU输出
        gru_out, hidden = self.gru(embedded)  # gru_out: [batch_size, seq_len, hidden_dim]
                                             # hidden: [num_layers, batch_size, hidden_dim]
        
        # 使用最后一个时间步的隐藏状态
        hidden = hidden[-1, :, :]  # [batch_size, hidden_dim]
        
        # 应用dropout
        hidden = self.dropout(hidden)
        
        # 全连接层
        out = self.fc(hidden)  # [batch_size, output_dim]
        return out

# 创建数据加载器
gru_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 模型参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 2  # 使用2层GRU

# 初始化模型
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), 'gru_text_classifier.pth')

# 文本分类函数
def classify_text_gru(text, model, char_to_index, max_len, index_to_label, device):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 创建标签索引映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型
new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label, device)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label, device)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

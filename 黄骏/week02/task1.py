import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()

        self.layers = nn.ModuleList()

        # 添加输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        # 添加额外的隐藏层
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())

        # 添加最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_model(hidden_dims, num_epochs=10, batch_size=32):
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)

    output_dim = len(label_to_index)
    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return epoch_losses, model


# 比较不同层数和节点数配置
configs = [
    {"name": "Single layer - 64 nodes", "hidden_dims": [64]},
    {"name": "Single layer - 128 nodes", "hidden_dims": [128]},
    {"name": "double layer - 64 nodes", "hidden_dims": [64, 64]},
    {"name": "double layer - 128 nodes", "hidden_dims": [128, 128]},
    {"name": "three layer - 64 nodes", "hidden_dims": [64, 64, 64]}
]

num_epochs = 10
results = {}

for config in configs:
    print(f"\n训练配置: {config['name']}")
    losses, model = train_model(config['hidden_dims'], num_epochs=num_epochs)
    results[config['name']] = losses

# 绘制不同配置的loss曲线
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(range(1, num_epochs + 1), losses, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss changes of different network structures')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison.png')
plt.show()

# 评估最终loss
print("\n各配置最终Loss:")
for name, losses in results.items():
    print(f"{name}: {losses[-1]:.4f}")

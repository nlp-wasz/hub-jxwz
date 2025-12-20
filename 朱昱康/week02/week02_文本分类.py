import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
vocab_size = len(char_to_index)
max_len = 40

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.labels = torch.tensor(labels, dtype=torch.long)
        bow_vectors = []
        for text in texts:
            tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
            tokenized += [0] * (max_len - len(tokenized))
            bow_vector = torch.zeros(vocab_size)
            for index in tokenized:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        self.bow_vectors = torch.stack(bow_vectors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

def build_mlp(input_dim, layer_sizes, output_dim):
    layers = []
    prev_dim = input_dim
    for hidden_dim in layer_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def train_and_record_loss(model, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
    return loss_history

layer_configs = [
    [16], [32], [64],
    [16, 16], [32, 32], [64, 64],
    [16, 16, 16], [32, 32, 32], [64, 64, 64]
]
output_dim = len(label_to_index)
num_epochs = 15

loss_curves = []
labels = []
layer_counts = []
neurons_per_layer = []
for config in layer_configs:
    print(f"Training structure: {config}")
    model = build_mlp(vocab_size, config, output_dim)
    loss_history = train_and_record_loss(model, dataloader, num_epochs=num_epochs)
    loss_curves.append(loss_history)
    labels.append(f"{len(config)} layers, {config[0]} neurons/layer")
    layer_counts.append(len(config))
    neurons_per_layer.append(config[0])

plt.figure(figsize=(12, 7))
for loss_history, label in zip(loss_curves, labels):
    plt.plot(range(1, num_epochs+1), loss_history, label=label)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison of Different Network Structures")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(200, 1) * 3.1415926
y_numpy = 2 * np.sin(X_numpy) + 1 + np.random.randn(200, 1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

x_train, x_test, y_train, y_test = X[:150, :], X[150:, :], y[:150, :], y[150:, :]
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)
print("数据生成完成。")
print("---" * 20)


class XSinDatasets(Dataset):
    def __init__(self, X,Y):
        self.x = X
        self.labels = Y
        self.max_len = len(X)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]





class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim,hidden_dim1, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

hidden_dim = 4
hidden_dim1 = 9
output_dim = 1
xSinDatasets = XSinDatasets(x_train,y_train)
dataloader = DataLoader(xSinDatasets, batch_size=10, shuffle=True)
model = SimpleClassifier(1, hidden_dim, hidden_dim1, output_dim)
# criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
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




# 3. 绘图：修复版本
model.eval()
with torch.no_grad():
    y_predicted = model(x_test)

# ✅ 关键：展平为 1D 数组
x_test_flat = x_test.numpy().squeeze()
y_test_flat = y_test.numpy().squeeze()
y_pred_flat = y_predicted.numpy().squeeze()

plt.figure(figsize=(100, 60))
plt.scatter(x_test_flat, y_test_flat, label='True Data', color='blue', alpha=1, s=300,linewidth=1)
plt.scatter(x_test_flat, y_pred_flat, label='Predicted', color='red', alpha=1, s=300,linewidth=1)

# # 可选：画出真实 sin 曲线对比
# x_line = np.linspace(0, np.pi, 100)
# y_line = 2 * np.sin(x_line) + 1
# plt.plot(x_line, y_line, 'g--', label='True sin(x) curve', alpha=0.8)

# 设置坐标轴标签
plt.xlabel('X', fontsize=12, loc='right')  # 靠右
plt.ylabel('y', fontsize=12, loc='top')    # 靠上
plt.title('True vs Predicted on Test Set')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
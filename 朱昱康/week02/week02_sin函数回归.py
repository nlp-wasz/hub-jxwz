import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Step1. 生成sin函数的数据\
# 训练数据
X_train_numpy = np.random.uniform(0, 2 * np.pi, (1000, 1))
y_train_numpy = np.sin(X_train_numpy) + 0.1 * np.random.randn(1000, 1)  # 加点噪声
X_train = torch.from_numpy(X_train_numpy).float()
y_train = torch.from_numpy(y_train_numpy).float()
# 测试数据
X_test_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_test_numpy = np.sin(X_test_numpy)
X_test = torch.from_numpy(X_test_numpy).float()
y_test = torch.from_numpy(y_test_numpy).float()
print("sin数据生成完成。")
print("-" * 30)


# Step2. 定义神经网络模型，损失函数和优化器
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)
model = NN()
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("-" * 30)


# Step3. 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print("\n训练完成！")
print("---" * 10)


# Step4. 测试并可视化
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).numpy()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_numpy, y_train_numpy, label='Train data', color='blue', alpha=0.5)
plt.plot(X_test_numpy, y_test_numpy, label='True sin(x)', color='green', linewidth=2)
plt.plot(X_test_numpy, y_pred_test, label='NN Prediction', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.title('NN拟合sin函数')
plt.grid(True)
plt.show()

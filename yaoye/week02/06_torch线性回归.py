import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 构建sin
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y_numpy = np.sin(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# 2. 构建多层神经网络
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
print(model)
print("---" * 10)

# 3. 损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.plot(X_numpy, y_numpy, label='True sin(x)', color='blue')
plt.plot(X_numpy, y_predicted, label='MLP Prediction', color='red')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.legend()
plt.grid(True)
plt.title('MLP fit sin(x)')

plt.show()

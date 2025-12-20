import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (改为sin函数)
X_numpy = np.random.rand(100, 1) * 2 * np.pi
# sin函数 + 噪音
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1  # y = sin(x) + 噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)


# 2. 创建多层网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 网络结构：1 -> 4 -> 4 -> 1
        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 替代原来的参数a和b
model = SimpleNet()

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000

for epoch in range(num_epochs):
    y_pred = model(X)  # 替代原来的 a * X + b

    # 计算损失（保持不变）
    loss = loss_fn(y_pred, y)

    # 反向传播和优化（保持不变）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 生成密集的测试点用于画平滑曲线
X_test_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
X_test = torch.from_numpy(X_test_numpy).float()

# 使用网络进行预测
model.eval()
# with torch.no_grad():
#     y_predicted_test = model(X_test).numpy()  # 测试点的预测

# 真实的sin函数（无噪音）
y_test_true = np.sin(X_test_numpy)


plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_test_numpy, y_test_true, 'r-', linewidth=2, label='sin(x)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

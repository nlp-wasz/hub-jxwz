import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成模拟数据 - sin函数
X_numpy = np.random.rand(1000, 1) * 4 * np.pi - 2 * np.pi
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X范围: [{X.min().item():.2f}, {X.max().item():.2f}]")
print("---" * 10)


# 2. 定义神经网络模型
class SinNet(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


# 创建模型实例
model = SinNet(hidden_size=64)
print("神经网络模型:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 2000
losses = []  # 记录损失值

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    losses.append(loss.item())

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 绘制训练损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 使用对数刻度更好地显示损失下降

# 6. 绘制拟合结果
plt.subplot(1, 2, 2)

# 生成测试数据点
X_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()

# 使用模型进行预测
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_true = np.sin(X_test)

# 绘制结果
plt.scatter(X_numpy, y_numpy, label='Noisy data', color='blue', alpha=0.3, s=10)
plt.plot(X_test, y_true, label='True sin(x)', color='green', linewidth=2)
plt.plot(X_test, y_pred, label='Model prediction', color='red', linewidth=2)
plt.title('Sin Function Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
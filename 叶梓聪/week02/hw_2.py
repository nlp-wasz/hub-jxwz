import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成Sin函数数据
X_numpy = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)  # 在[-π, π]区间均匀采样
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数+噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"生成 {len(X)} 个数据点")
print("---" * 10)


# 2. 定义神经网络模型
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 128),  # 输入层 -> 隐藏层1
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 64),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 32),  # 隐藏层2 -> 隐藏层3
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 1)  # 隐藏层3 -> 输出层
        )

    def forward(self, x):
        return self.network(x)


# 初始化模型、损失函数和优化器
model = SinModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("模型架构:")
print(model)
print("---" * 10)

# 3. 训练模型
num_epochs = 2000
losses = []  # 用于记录损失值变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 4. 绘制训练过程
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 5. 绘制拟合结果
with torch.no_grad():
    predicted = model(X).numpy()

plt.subplot(1, 2, 2)
plt.scatter(X_numpy, y_numpy, label='Noisy Sin Data', color='blue', alpha=0.4, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='True Sin Function', color='green', linewidth=2)
plt.plot(X_numpy, predicted, label='Model Prediction', color='red', linewidth=2)
plt.legend()
plt.title('Sin Function Fitting')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n训练完成！")
print("---" * 10)

# 6. 评估模型
test_X = torch.linspace(-3, 3, 600).reshape(-1, 1).float()
with torch.no_grad():
    test_y = model(test_X).numpy()

plt.figure(figsize=(10, 6))
plt.plot(test_X.numpy(), test_y, label='Model Extrapolation', color='red', linewidth=2)
plt.plot(test_X.numpy(), np.sin(test_X.numpy()), label='True Sin Function', color='green', linewidth=2)
plt.axvline(x=np.pi, color='gray', linestyle='--')
plt.axvline(x=-np.pi, color='gray', linestyle='--')
plt.text(-3.5, 0, 'Training Range', fontsize=12)
plt.title('Extrapolation Test (Beyond Training Range)')
plt.legend()
plt.grid(True)
plt.show()

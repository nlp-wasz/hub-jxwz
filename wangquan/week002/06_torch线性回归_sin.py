import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy)

# 添加一些噪声使任务更具挑战性
y_numpy += 0.1 * np.random.randn(1000, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print("---" * 10)


# 2. 定义多层神经网络
class SinRegressionModel(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(SinRegressionModel, self).__init__()
        self.layers = torch.nn.ModuleList()

        # 输入层
        self.layers.append(torch.nn.Linear(1, hidden_size))
        self.layers.append(torch.nn.ReLU())

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())

        # 输出层
        self.layers.append(torch.nn.Linear(hidden_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 初始化模型
model = SinRegressionModel(hidden_size=64, num_layers=4)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
losses = []

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

# 5. 绘制训练损失
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# 6. 绘制拟合结果
plt.subplot(1, 2, 2)
# 生成更密集的点用于绘制平滑曲线
X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
X_plot_tensor = torch.from_numpy(X_plot).float()

# 使用模型进行预测
model.eval()
with torch.no_grad():
    y_pred_plot = model(X_plot_tensor).numpy()

plt.scatter(X_numpy, y_numpy, label='Noisy sin(x)', color='blue', alpha=0.3, s=5)
plt.plot(X_plot, np.sin(X_plot), label='True sin(x)', color='green', linewidth=2)
plt.plot(X_plot, y_pred_plot, label='Model prediction', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Sin Function Fitting with Neural Network')

plt.tight_layout()
plt.show()

# 打印最终损失
print(f'Final Loss: {losses[-1]:.6f}')

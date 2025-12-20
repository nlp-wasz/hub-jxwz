import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 生成sin函数模拟数据
np.random.seed(42)
torch.manual_seed(42)

X_numpy = np.random.uniform(-2 * np.pi, 2 * np.pi, (1000, 1))
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print("Sin函数数据生成完成。")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print("---" * 10)


# 2. 定义多层神经网络模型
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


# 3. 初始化模型、损失函数和优化器
model = SinNet(hidden_size=128)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("模型结构:")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 2000
train_losses = []
test_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    # 训练模式
    model.train()

    # 前向传播
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 评估模式
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 绘制训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(test_losses, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

# 6. 绘制拟合结果
plt.subplot(1, 2, 2)
# 生成用于绘图的密集点
x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
x_plot_tensor = torch.from_numpy(x_plot).float()

model.eval()
with torch.no_grad():
    y_plot_pred = model(x_plot_tensor).numpy()

plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.6, s=10)
plt.scatter(X_test, y_test, label='Test Data', color='green', alpha=0.6, s=10)
plt.plot(x_plot, np.sin(x_plot), label='True sin(x)', color='black', linewidth=2, linestyle='--')
plt.plot(x_plot, y_plot_pred, label='Neural Network Fit', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin Function Fitting')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 在测试集上评估最终性能
model.eval()
with torch.no_grad():
    final_test_pred = model(X_test_tensor)
    final_test_loss = criterion(final_test_pred, y_test_tensor)
    print(f"最终测试集损失: {final_test_loss.item():.6f}")

# 8. 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数数量: {total_params}")

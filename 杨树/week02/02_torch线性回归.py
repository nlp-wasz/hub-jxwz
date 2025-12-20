import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

# 1. 生成sin函数数据
np.random.seed(42)  # 设置随机种子以确保可重复性
X_numpy = np.random.rand(1000, 1) * 4 * np.pi - 2 * np.pi  # x范围: [-2π, 2π]
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin(x) + 噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print(f"数据形状: X: {X.shape}, y: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self, hidden_layers=[64, 32, 16]):
        super(SinNet, self).__init__()
        layers = []

        # 输入层 (1个特征 -> 第一个隐藏层)
        layers.append(torch.nn.Linear(1, hidden_layers[0]))
        layers.append(torch.nn.ReLU())

        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(torch.nn.ReLU())

        # 输出层 (最后一个隐藏层 -> 1个输出)
        layers.append(torch.nn.Linear(hidden_layers[-1], 1))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 创建模型
model = SinNet(hidden_layers=[128, 64, 32, 16])
print("模型结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 4. 训练模型
num_epochs = 5000
losses = []

print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 每500个epoch打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    # 生成测试数据用于绘制平滑曲线
    X_test = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1).float()
    y_test_true = torch.sin(X_test)
    y_test_pred = model(X_test)

# 6. 绘制结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 子图1: 拟合效果
ax1.scatter(X_numpy, y_numpy, label='训练数据 (含噪声)', color='blue', alpha=0.6, s=10)
ax1.plot(X_test, y_test_true, label='真实 sin(x)', color='green', linewidth=3, linestyle='--')
ax1.plot(X_test, y_test_pred, label='神经网络拟合', color='red', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Sin函数拟合效果')
ax1.legend()
ax1.grid(True)

# 子图2: 训练损失曲线
ax2.plot(losses, label='训练损失', color='purple')
ax2.set_xlabel('训练轮次')
ax2.set_ylabel('损失值 (MSE)')
ax2.set_title('训练损失曲线')
ax2.set_yscale('log')  # 使用对数坐标更好地显示损失下降
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 7. 计算拟合误差
mse = torch.nn.functional.mse_loss(y_test_pred, y_test_true).item()
print(f"测试集MSE误差: {mse:.6f}")

# 8. 在几个点上测试预测
test_points = torch.tensor([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]).float().reshape(-1, 1)
with torch.no_grad():
    predictions = model(test_points)
    true_values = torch.sin(test_points)

print("\n测试点预测结果:")
print(" x\t\t真实值\t\t预测值\t\t误差")
print("-" * 50)
for i in range(len(test_points)):
    x_val = test_points[i].item()
    true_val = true_values[i].item()
    pred_val = predictions[i].item()
    error = abs(true_val - pred_val)
    print(f"{x_val:6.3f}\t{true_val:8.4f}\t{pred_val:8.4f}\t{error:8.4f}")
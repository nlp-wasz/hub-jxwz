import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成sin函数数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 生成-2π到2π之间的1000个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上一些噪声

print(f"数据生成完成，X范围: [{X_numpy.min():.2f}, {X_numpy.max():.2f}]")
print(f"数据点数: {len(X_numpy)}")
print("---" * 10)

# 2. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()


# 3. 定义多层神经网络模型
class SinRegressionModel(nn.Module):
    def __init__(self, hidden_layers=[64, 32, 16]):
        super(SinRegressionModel, self).__init__()

        layers = []
        input_dim = 1  # 输入维度是1（x值）

        # 构建隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 轻微dropout防止过拟合
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 4. 初始化模型、损失函数和优化器
model = SinRegressionModel(hidden_layers=[128, 64, 32, 16])  # 4层隐藏层
criterion = nn.MSELoss()  # 均方误差损失，适合回归问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam优化器

print(f"模型结构: {model}")
print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
print("---" * 10)

# 5. 训练模型
num_epochs = 2000
train_losses = []

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
        # 训练集损失
        train_loss = loss.item()
        train_losses.append(train_loss)

    # 每200个epoch打印一次进度
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}')

print("\n训练完成！")
print("---" * 10)

# 6. 绘制训练过程
plt.figure(figsize=(15, 10))

# 子图1: 训练和测试损失曲线
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')  # 使用对数坐标更好地观察损失下降

# 子图2: 最终拟合结果
plt.subplot(2, 1, 2)
model.eval()
with torch.no_grad():
    # 生成平滑的预测曲线
    X_smooth = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    X_smooth_tensor = torch.from_numpy(X_smooth).float()
    y_smooth_pred = model(X_smooth_tensor).numpy()

    # 真实sin函数（无噪声）
    y_true = np.sin(X_smooth)

    plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.6, s=10)
    plt.scatter(X_test, y_test, label='Test Data', color='green', alpha=0.6, s=10)
    plt.plot(X_smooth, y_true, label='True sin(x)', color='black', linewidth=3, linestyle='--')
    plt.plot(X_smooth, y_smooth_pred, label='Model Prediction', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sin Function Fitting')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
# 7. 打印最终评估结果
model.eval()
with torch.no_grad():
    final_train_loss = criterion(model(X_train_tensor), y_train_tensor).item()
    final_test_loss = criterion(model(X_test_tensor), y_test_tensor).item()

print(f"最终训练损失: {final_train_loss:.6f}")
print(f"最终测试损失: {final_test_loss:.6f}")
print(f"测试集RMSE: {np.sqrt(final_test_loss):.6f}")


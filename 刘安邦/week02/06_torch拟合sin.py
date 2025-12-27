"""
调整 06_torch线性回归.py 构建一个sin函数，
然后通过多层网络拟合sin函数，并进行可视化。
"""
import torch
import torch.nn as nn
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

# y_numpy = np.sin(X_numpy)
y_numpy = np.sin(X_numpy) - 0.5 + np.random.randn(100, 1) * 0.1

X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 创建多层网络
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),  # 增加模型的复杂度，非线性

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 输出层：从 hidden_size2 到 output_size
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.network(x)


model = MLP(input_size=1, hidden_size1=32, hidden_size2=32, output_size=1)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 回归任务

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model.forward(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 绘制结果
# 使用最终学到的参数 来计算拟合直线的 y 值
with torch.no_grad():
    X_test = np.linspace(0, 10, 500).reshape(-1, 1)  # 500个均匀分布的点
    X_test_tensor = torch.from_numpy(X_test).float()
    y_pred_smooth = model(X_test_tensor)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_test, y_pred_smooth, label='Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

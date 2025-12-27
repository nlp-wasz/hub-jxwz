import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成sin模拟数据
X = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(100, 1))
y = np.sin(X) + 0.05 * np.random.randn(100, 1)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

print("数据生成完成。")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # 输入层到隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 64),  # 隐藏层到隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 1)  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.net(x)


# 3.创建模型实例
model = SinModel()

# 4. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 设置训练模型
    model.train()
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a = model.net[0].weight[0][0]
b = model.net[0].bias[0]
print("参数 a:", a, "参数 b:", b)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    model.eval()
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X, y_predicted, label=f'Model: y = {a:.2f}x + {b:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

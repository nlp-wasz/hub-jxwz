import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 0到2π之间的1000个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 加入少量噪声
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成 (sin函数 + 噪声)")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinApproximator(nn.Module):
    def __init__(self):
        super(SinApproximator, self).__init__()
        # 定义多层感知机结构
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # 输入层：1维到64维
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 128),  # 隐藏层：64维到128维
            nn.Tanh(),  # 激活函数
            nn.Linear(128, 64),  # 隐藏层：128维到64维
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 1)  # 输出层：64维到1维
        )

    def forward(self, x):
        return self.layers(x)


# 3. 初始化模型、损失函数和优化器
model = SinApproximator()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 模型预测
with torch.no_grad():
    y_pred = model(X)
    y_pred_numpy = y_pred.numpy()
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei",  "Microsoft YaHei"]
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 6. 绘制结果
plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label='带噪声的sin数据', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='原始sin函数', color='green', linewidth=2)
plt.plot(X_numpy, y_pred_numpy, label='神经网络拟合曲线', color='red', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('多层神经网络拟合sin函数')
plt.legend()
plt.grid(True)
plt.show()

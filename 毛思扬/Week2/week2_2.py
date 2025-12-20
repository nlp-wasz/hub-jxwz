# 调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn

# macOS系统中常用的中文字体
zh_font = fm.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']


# 构建数据集
def get_data(X):
    c, r = X.shape
    y = np.sin(X * 3.14) + 1 + (0.02 * (2 * np.random.rand(c, r) - 1))
    return y


# 生成数据
X_numpy = np.arange(0, 10, 0.01).reshape(-1, 1)
y_numpy = get_data(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 定义多层神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(1, 64)  # 输入层到第一隐藏层
        self.hidden2 = nn.Linear(64, 32)  # 第一隐藏层到第二隐藏层
        self.hidden3 = nn.Linear(32, 16)  # 第二隐藏层到第三隐藏层
        self.hidden4 = nn.Linear(16, 8)  # 第三隐藏层到第四隐藏层
        self.output = nn.Linear(8, 1)  # 第四隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # 使用ReLU激活函数
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = self.output(x)
        return x


# 创建模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("开始训练...")
print("---" * 10)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 使用训练好的模型进行预测
with torch.no_grad():
    y_predicted = model(X)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label='原始数据', color='blue', alpha=0.6, s=10)
plt.plot(X_numpy, y_predicted.numpy(), label='神经网络拟合结果', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('使用多层神经网络拟合sin函数')
plt.show()

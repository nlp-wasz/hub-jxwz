import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
y_numpy = np.sin(X_numpy)#  + np.random.randn(100, 1)*0.2
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),

            # 输出层：从 hidden_size3 到 output_size
            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.network(x)


# --- 模型参数和实例化 ---
input_size = 1
hidden_size1 = 20
hidden_size2 = 30
hidden_size3 = 40
output_size = 1

# 实例化模型
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# 定义损失函数 (均方误差)
loss_fn = nn.MSELoss()

# 定义优化器 (随机梯度下降)
# model.parameters() 会自动找到模型中需要优化的参数（即a和b）
optimizer = torch.optim.Adam(model.parameters(), lr=0.02) # lr 是学习率

# 训练模型
num_epochs = 1000  # 训练迭代次数
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化：
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印最终学到的参数
# model.weight 是斜率a， model.bias 是截距b
print("\n训练完成！")

# 将模型切换到评估模式，这在训练结束后是好习惯
model.eval()

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted = model(X).numpy() # 使用训练好的模型进行预测

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_numpy, y_predicted, label='test data', color='red', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

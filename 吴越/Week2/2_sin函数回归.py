import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

# 1. 生成1000行1列的随机矩阵，数值在[-2π,2π]区间均匀分布
x_train_numpy = np.random.uniform(-2*np.pi,2*np.pi,(1000,1))
y_train_numpy= np.sin(x_train_numpy) + 0.1*np.random.randn(1000, 1)#添加噪声
# 将NumPy数组转换为tensor张量
x = torch.from_numpy(x_train_numpy).float()
y = torch.from_numpy(y_train_numpy).float()

# 测试数据
X_test_numpy = np.linspace(-2*np.pi, 2 * np.pi, 500).reshape(-1, 1)
y_test_numpy = np.sin(X_test_numpy)
X_test = torch.from_numpy(X_test_numpy).float()
y_test = torch.from_numpy(y_test_numpy).float()

#2.定义多层线性模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3,output_size):
        super(MLP, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(), # 增加模型的复杂度，非线性

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

model=MLP(1,32,64,32,1)
# #3.定义损失函数
loss_fn = nn.MSELoss()
# #4.定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam 优化器  可以结合梯度 动态调整学习， 0.01 -> 0.001 -> 0.00001
# #5.模型训练
num_epochs = 1000 # 训练迭代次数
for epoch in range(num_epochs):
     # 前向传播：计算预测值
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化：
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



print("\n训练完成！")

# 将模型切换到评估模式
model.eval()

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
     y_predicted = model(X_test).numpy() # 使用训练好的模型进行预测

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x_train_numpy, y_train_numpy, label='Train data',color='blue',alpha=0.5)
plt.plot(X_test_numpy, y_test_numpy, label='True sin(x)', color='green',linewidth=2)
plt.plot(X_test_numpy, y_predicted, label='Model Prediction',color='red',linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
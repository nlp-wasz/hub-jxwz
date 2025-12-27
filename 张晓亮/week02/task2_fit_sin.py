from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim

# 1. 生成模拟数据 - 修改为与测试数据相同的范围
X_numpy = np.random.uniform(-2 * np.pi, 2 * np.pi, (100, 1))
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)  # 减少噪声强度
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dim: List[int] = [128, 16],
                 activation: str = "relu"):
        super(SimpleClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.model = self.build_model()

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.activation == "tanh":
            return nn.Tanh()  # 添加tanh激活函数，更适合拟合sin函数

    def build_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        layers.append(self.get_activation())

        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            layers.append(self.get_activation())

        # 输出层不使用激活函数，因为我们要拟合任意实数值
        layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, x_train, y_train, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}')

    return train_losses


# 训练模型 - 使用tanh激活函数，更适合拟合sin函数
model = SimpleClassifier(hidden_dim=[64, 16], activation="tanh")
train_losses = train_model(model, X, y)

# 创建测试数据
x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_test = np.sin(x_test)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    x_test_t = torch.FloatTensor(x_test)
    y_pred = model(x_test_t).numpy()

# 绘制结果
plt.figure(figsize=(15, 5))

# 子图1: 真实函数与预测函数
# plt.subplot(1, 3, 1)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='Training Data', s=10)
plt.plot(x_test, y_test, 'r-', label='True Function', linewidth=2)
plt.plot(x_test, y_pred, 'g-', label='Model Prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()  # 添加这行代码来显示图形
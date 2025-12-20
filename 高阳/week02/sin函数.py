import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 生成训练数据
x = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).view(-1, 1)
y = torch.sin(x)


# 定义神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



# 初始化模型和优化器
model = SimpleClassifier(1, 128, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00165)

# 训练过程
plt.ion()
for epoch in range(5000):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy(), c='blue', label='True')
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=3, label='Predicted')
        plt.text(0, -0.8, 'Epoch=%d, Loss=%.4f' % (epoch, loss.item()))
        plt.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()

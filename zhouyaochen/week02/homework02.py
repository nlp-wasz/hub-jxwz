import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""
#1.模拟数据
x_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_numpy = np.sin(x_numpy)+0.1*np.random.randn(*x_numpy.shape)

x = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()


#2.定义多层网络
class SinFunction(nn.Module):
    def __init__(self,input_dim=1,hidden_dim=64,output_dim=1):
        super(SinFunction, self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


model = SinFunction()

#3.定义损失函数和优化器 这里用Adam
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#4.训练
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    #每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#可视化结果
model.eval()
with torch.no_grad():
    output = model(x)

plt.figure(figsize=(10,6))
plt.scatter(x_numpy, y_numpy, color='blue', alpha=0.6, label='Noisy Data')
plt.plot(x_numpy, output.numpy(), color='red', linewidth=2, label='NN Prediction')
plt.plot(x_numpy, np.sin(x_numpy), color='green', linestyle='--', label='True sin(x)')
plt.legend()
plt.grid(True)
plt.show()
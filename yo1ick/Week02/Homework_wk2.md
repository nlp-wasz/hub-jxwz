# 第二周作业
## 调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化
以原代码运行结果
> Epoch [10/10], Loss: 0.5861

> 输入 '帮我导航到北京' 预测为: 'Travel-Query'
> 输入 '查询明天北京的天气' 预测为: 'Weather-Query'

修改SimpleClassifier类
```python
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
```
同时增加节点数量
```python
hidden_dim = 256
hidden_dim_2 = 128
```
修改后结果：
>Epoch [10/10], Loss: 0.6553

>输入 '帮我导航到北京' 预测为: 'Travel-Query'
>输入 '查询明天北京的天气' 预测为: 'Weather-Query'

## 调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
```python
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # 添加一些噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# 2. 构建多层网络模型
class SinModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(SinModel, self).__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 3. 定义模型、损失函数和优化器
model = SinModel(hidden_size=64, num_layers=4)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")

# 5. 绘制训练损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 使用对数坐标更好地观察损失变化

# 6. 绘制拟合结果
plt.subplot(1, 2, 2)
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.scatter(X_numpy, y_numpy, label='Noisy sin data', color='blue', alpha=0.3, s=5)
plt.plot(X_numpy, y_predicted, label='Model prediction', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin function', color='green', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Sin Function Fitting')

plt.tight_layout()
plt.show()
```

训练结果：
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/7500bcc2-4ece-44bd-af64-81d94772a48c" />

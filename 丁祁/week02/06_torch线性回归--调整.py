import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成模拟数据 (与之前相同)
X_numpy =np.linspace(-np.pi, np.pi, 100)  # 生成100个点，从-π到π
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy)+ 0.1 * np.random.randn(100)
X = torch.from_numpy(X_numpy).float().view(-1, 1) # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float().view(-1, 1)

print("数据生成完成。")
print("---" * 10)

loss_fn = torch.nn.MSELoss() # 回归任务

def train_math(train_model):
    train_optimizer = optim.SGD(train_model.parameters(), lr=0.01)
    num_epochs = 1000
    for epoch in range(num_epochs):
        outputs = train_model(X)
        loss = loss_fn(outputs, y)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if epoch % 50 == 0:
            print(f"Batch 个数 {epoch}, 当前Batch Loss: {loss.item()}")


train_model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
train_math(train_model)
train_model.eval()


with torch.no_grad():
    y_predicted = train_model(X)


plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y =sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

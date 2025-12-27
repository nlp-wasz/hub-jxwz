import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据（基于 sin 函数）
# 随机生成均匀分布的 [0, 1) 浮点数，其中括号内容是目标形状，维度任意
X_numpy = np.random.rand(100, 1) * 2 * np.pi  # 在 [0, 2π] 区间内生成数据
# 生成符合正态分布的(-∞, +∞)区间随机数
X_numpy_ = np.random.randn(1000, 1)
# 生成 [1, 10) 的整数
X_numpy_ = np.random.randint(1, 10, 10)

y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1  # y = sin(x) + 噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义参数
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([a, b], lr=0.001)  # 降低学习率以适应非线性

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播：y = a * sin(x) + b
    y_pred = a * torch.sin(X) + b

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# with torch.no_grad():
#     y_predicted = a_learned * torch.sin(X) + b_learned

# plt.figure(figsize=(10, 6))
# plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# plt.plot(X_numpy, y_predicted.numpy(), label=f'Model: y = {a_learned:.2f}sin(x) + {b_learned:.2f}', color='red', linewidth=2)
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()
# 生成排序后的 X 值，用于绘制平滑曲线
X_plot = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
X_plot_tensor = torch.from_numpy(X_plot).float()

# 使用最终参数进行预测
with torch.no_grad():
    y_predicted = a_learned * torch.sin(X_plot_tensor) + b_learned

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_plot, y_predicted.numpy(), label=f'Model: y = {a_learned:.2f}sin(x) + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


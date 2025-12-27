import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import torch
import numpy as np


# 1. 生成模拟数据 (基于正弦函数)
# 创建0到5之间的100个点
X_numpy = np.linspace(0, 5, 100).reshape(-1, 1)
# 生成带噪声的正弦函数数据: y = 2*sin(3x + 1) + 0.5 + 噪声
y_numpy = 2 * np.sin(3 * X_numpy) + 0.5 + 0.0001 * np.random.randn(100, 1)

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 创建参数张量 A, B, C
# 正弦函数形式: y = A*sin(Bx) + C
A = torch.randn(1, requires_grad=True, dtype=torch.float)
B = torch.randn(1, requires_grad=True, dtype=torch.float)
C = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 A: {A.item():.4f}")
print(f"初始参数 B: {B.item():.4f}")
print(f"初始参数 C: {C.item():.4f}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam([A, B, C], lr=0.01)  # 使用Adam优化器可能收敛更快

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播：计算 y_pred = A*sin(Bx) + C
    y_pred = A * torch.sin(B * X) + C

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每500个 epoch 打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
A_learned = A.item()
B_learned = B.item()
C_learned = C.item()
print(f"拟合的参数 A: {A_learned:.4f}")
print(f"拟合的参数 B: {B_learned:.4f}")
print(f"拟合的参数 C: {C_learned:.4f}")
print("原始参数为 A=2, B=3, C=1")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    y_predicted = A_learned * np.sin(B_learned * X_numpy) + C_learned

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Fitted: y = {A_learned:.2f}*sin({B_learned:.2f}x) + {C_learned:.2f}',
         color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Sine Function Fitting with PyTorch')
plt.show()

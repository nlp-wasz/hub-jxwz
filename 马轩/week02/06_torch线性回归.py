import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 生成数据集
X_numpy = np.linspace(0, 2*np.pi, 1000)
# 真实模型: y = sin(4*x) + 3 + 噪声
y_numpy = np.sin(4*X_numpy) + 3 + np.random.randn(1000) * 0.1

# 转换为PyTorch张量并移动到指定设备 使用 unsqueeze(1) 后，形状变为 (1000, 1)
X = torch.from_numpy(X_numpy).float().unsqueeze(1).to(device)  # 增加维度并移至GPU/CPU
y = torch.from_numpy(y_numpy).float().unsqueeze(1).to(device)

# 2. 定义多层神经网络模型
class DeepSinModel(nn.Module):
    def __init__(self):
        super(DeepSinModel, self).__init__()
        # 定义多层神经网络
        self.layers = nn.Sequential(
            nn.Linear(1, 64),    # 输入层: 1个特征 -> 64个神经元
            nn.Tanh(),           # 激活函数
            nn.Linear(64, 128),  # 隐藏层: 64 -> 128
            nn.Tanh(),           # 激活函数
            nn.Linear(128, 64),  # 隐藏层: 128 -> 64
            nn.Tanh(),           # 激活函数
            nn.Linear(64, 1)     # 输出层: 64 -> 1 (预测单个值)
        )
        
    def forward(self, x):
        # 前向传播
        return self.layers(x)

# 3. 初始化模型、损失函数和优化器
model = DeepSinModel().to(device)
criterion = nn.MSELoss()  # 回归问题使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 4. 训练模型
epochs = 5000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    # 每1000个epoch打印一次信息
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 5. 可视化结果（需要将数据移回CPU）
plt.figure(figsize=(10, 6))
plt.scatter(X.cpu().numpy(), y.cpu().numpy(), s=1, label='带噪声的训练数据')
plt.plot(X.cpu().numpy(), model(X).cpu().detach().numpy(), 'r-', linewidth=2, label='模型拟合曲线')
plt.plot(X_numpy, np.sin(4*X_numpy) + 3, 'g--', linewidth=2, label='真实曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('多层神经网络拟合正弦曲线结果')
plt.legend()
plt.show()
    

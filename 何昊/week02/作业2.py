import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成带噪声的正弦函数数据
def generate_sine_data(n_samples=1000, noise_level=0.1):
    # 生成输入数据
    X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    
    # 生成纯净的正弦值
    y_clean = np.sin(X_numpy)
    
    # 添加高斯噪声
    y_noisy = y_clean + noise_level * np.random.randn(n_samples, 1)
    
    return X_numpy, y_clean, y_noisy

# 生成数据
X_numpy, y_clean, y_noisy = generate_sine_data(n_samples=1000, noise_level=0.1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_numpy, y_noisy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# 2. 定义多层感知机模型
class SineNet(nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(SineNet, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# 3. 训练模型
def train_model(model, X_train, y_train, X_test, y_test, num_epochs=1000, learning_rate=0.01):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练过程中的损失
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            train_loss = criterion(train_pred, y_train)
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        
        # 记录损失
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # 每100个epoch打印一次进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')
    
    return train_losses, test_losses

# 创建模型实例
model = SineNet(hidden_size=64, num_layers=4)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 训练模型
train_losses, test_losses = train_model(
    model, 
    X_train_tensor, 
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    num_epochs=2000,
    learning_rate=0.001
)

# 4. 可视化结果
def plot_results(model, X_numpy, y_clean, y_noisy):
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        X_full = torch.from_numpy(X_numpy).float()
        y_pred = model(X_full).numpy()
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # 使用对数尺度更好地显示损失变化
    plt.legend()
    plt.title('Training and Test Loss')
    plt.grid(True)
    
    # 绘制拟合结果
    plt.subplot(2, 2, 2)
    plt.scatter(X_numpy, y_noisy, alpha=0.3, label='Noisy Data', s=10, color='blue')
    plt.plot(X_numpy, y_clean, label='True Sine Function', linewidth=2, color='green')
    plt.plot(X_numpy, y_pred, label='MLP Prediction', linewidth=2, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Sine Function Fitting')
    plt.grid(True)
    
    # 绘制误差分布
    plt.subplot(2, 2, 3)
    errors = y_pred - y_clean
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    # 绘制预测值与真实值的关系
    plt.subplot(2, 2, 4)
    plt.scatter(y_clean, y_pred, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # 理想预测线
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 显示结果
plot_results(model, X_numpy, y_clean, y_noisy)

# 5. 评估最终性能
model.eval()
with torch.no_grad():
    # 在整个数据集上评估
    X_full = torch.from_numpy(X_numpy).float()
    y_pred = model(X_full)
    
    # 计算各种指标
    mse = nn.MSELoss()(y_pred, torch.from_numpy(y_clean).float()).item()
    mae = nn.L1Loss()(y_pred, torch.from_numpy(y_clean).float()).item()
    
    print(f"\n最终性能指标:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {np.sqrt(mse):.6f}")
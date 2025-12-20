# Week02/sin_fit6.py
# -*- coding: utf-8 -*-
"""
通过神经网络拟合正弦函数（超参数优化版）
1. 测试不同层数和激活函数的组合
2. 跟踪MSE变化寻找最佳组合
3. 保存最佳组合下拟合效果图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------- 实验配置 --------------------
class Config:
    experiment_dir = "experiments"  # 实验结果的保存目录

os.makedirs(Config.experiment_dir, exist_ok=True)

# 设置matplotlib为非交互模式，图像显示后继续执行
plt.ioff()

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 1. 生成带噪声的sin函数数据集
def generate_sin_data(n_samples=1000, noise_level=0.1):
    # 生成输入值
    x = np.linspace(-2*np.pi, 2*np.pi, n_samples)
    # 计算对应的sin值
    y_true = np.sin(x)
    # 添加高斯噪声
    y_noisy = y_true + np.random.normal(0, noise_level, n_samples)
    return x, y_true, y_noisy

# 生成数据
x, y_true, y_noisy = generate_sin_data(1000, 0.1)

# 2. 显示数据生成后的图片
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(x, y_true, label='True sin(x)', linewidth=3, color='blue')
plt.title('True Sin Function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x, y_noisy, label='Noisy data', s=5, alpha=0.6, color='red')
plt.plot(x, y_true, label='True sin(x)', linewidth=2, color='blue')
plt.title('Noisy Sin Function Data')
plt.xlabel('x')
plt.ylabel('sin(x) + noise')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 3)
# 显示边界区域的问题区域
plt.plot(x, y_true, label='True sin(x)', linewidth=3, color='blue')
plt.scatter(x, y_noisy, label='Noisy data', s=3, alpha=0.4, color='red')
# 标记边界问题区域
left_boundary_mask = (x >= -2*np.pi) & (x <= -3.5)
right_boundary_mask = (x >= 3.5) & (x <= 2*np.pi)
plt.fill_between(x[left_boundary_mask], -1.5, 1.5, alpha=0.2, color='orange', label='Problem Area (Left)')
plt.fill_between(x[right_boundary_mask], -1.5, 1.5, alpha=0.2, color='orange', label='Problem Area (Right)')
plt.title('Problem Boundary Areas Highlighted')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 4)
# 数据分布直方图
plt.hist(y_noisy, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Noisy Data')
plt.xlabel('sin(x) + noise')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.experiment_dir, 'datageneration_analysis.png'), dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(2)
print("数据生成完成！已显示数据分布和分析图。")

# 3. 改进的多层神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # 使用Xavier/Glorot初始化
        self.weights = []
        self.biases = []
        
        # 输入层到第一个隐藏层
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # 隐藏层之间
        for i in range(len(hidden_sizes)-1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * np.sqrt(2.0 / hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
        
        # 输出层
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def activation(self, x, activation_type='tanh'):
        if activation_type == 'sigmoid':
            return self.sigmoid(x)
        elif activation_type == 'tanh':
            return self.tanh(x)
        elif activation_type == 'relu':
            return self.relu(x)
        elif activation_type == 'leaky_relu':
            return self.leaky_relu(x)
        return x
    
    def activation_derivative(self, x, activation_type='tanh'):
        if activation_type == 'sigmoid':
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        elif activation_type == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        return 1
    
    def forward(self, x, activation_type='tanh'):
        self.activations = [x]
        self.z_values = []
        
        # 前向传播
        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.activation(z, activation_type))
        
        # 输出层（线性激活）
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)
        
        return self.activations[-1]
    
    def backward(self, x, y, learning_rate=0.01, activation_type='tanh'):
        m = x.shape[0]
        
        # 前向传播
        output = self.forward(x, activation_type)
        
        # 计算输出层的误差
        dZ = output - y
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients = [(dW, db)]
        
        # 反向传播
        for i in range(len(self.weights)-2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.activation_derivative(self.z_values[i], activation_type)
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))
        
        # 更新权重和偏置
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]
    
    def train(self, x, y, epochs=1000, learning_rate=0.01, activation_type='tanh', verbose=True):
        losses = []
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(epochs):
            # 动态学习率衰减
            current_lr = learning_rate * (0.95 ** (epoch // 200))
            
            # 前向传播
            output = self.forward(x, activation_type)
            
            # 计算损失
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # 早停机制
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # 反向传播
            self.backward(x, y, current_lr, activation_type)
            
            if verbose and epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr:.6f}')
        
        return losses

# 4. 准备训练数据
X_train = x.reshape(-1, 1)
y_train = y_noisy.reshape(-1, 1)

# 5. 寻找最佳网络结构和激活函数
print("开始寻找最佳网络结构和激活函数...")

# 测试不同的网络结构
hidden_configs = [
    [32, 16],
    [64, 32],
    [128, 64, 32],
    [64, 32, 16, 8],
    [256, 128, 64]
]

# 测试不同的激活函数
activation_types = ['tanh', 'sigmoid', 'relu', 'leaky_relu']

# 初始化最佳参数（使用默认值，避免None）
best_mse = float('inf')
best_structure = [64, 32]  # 默认值
best_activation = 'tanh'   # 默认值
best_model = None
best_predictions = None
results = {}

start_time = time.time()

for i, hidden_sizes in enumerate(hidden_configs):
    for j, act in enumerate(activation_types):
        print(f"\n测试 结构{hidden_sizes} + 激活函数{act} ({i+1}/{len(hidden_configs)}, {j+1}/{len(activation_types)})")
        
        nn = NeuralNetwork(input_size=1, hidden_sizes=hidden_sizes, output_size=1)
        losses = nn.train(X_train, y_train, epochs=2000, learning_rate=0.01, activation_type=act, verbose=False)
        
        predictions = nn.forward(X_train, act)
        mse = np.mean((predictions - y_true.reshape(-1, 1)) ** 2)
        
        key = f"结构{hidden_sizes}_激活函数{act}"
        results[key] = mse
        
        if mse < best_mse:
            best_mse = mse
            best_structure = hidden_sizes
            best_activation = act
            best_model = nn
            best_predictions = predictions
        
        print(f"MSE: {mse:.6f}")

end_time = time.time()
print(f"\n超参数搜索完成!耗时: {end_time - start_time:.2f}秒")

# 6. 使用最佳模型进行最终训练
print(f"\n最佳配置: 结构{best_structure}, 激活函数{best_activation}, MSE: {best_mse:.6f}")
print("使用最佳配置进行最终训练...")

# 确保best_model存在，如果不存在则重新创建
if best_model is None:
    print("重新创建最佳模型...")
    best_model = NeuralNetwork(input_size=1, hidden_sizes=best_structure, output_size=1)

final_losses = best_model.train(X_train, y_train, epochs=6000, learning_rate=0.008, activation_type=best_activation, verbose=True)

final_predictions = best_model.forward(X_train, best_activation)
final_mse = np.mean((final_predictions - y_true.reshape(-1, 1)) ** 2)

# 7. 计算评估指标
y_true_reshaped = y_true.reshape(-1, 1)
mse = np.mean((final_predictions - y_true_reshaped) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(final_predictions - y_true_reshaped))
ss_res = np.sum((y_true_reshaped - final_predictions) ** 2)
ss_tot = np.sum((y_true_reshaped - np.mean(y_true_reshaped)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

print(f"\n最终模型评估指标:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.6f}")

# 8. 生成最佳结果图
plt.figure(figsize=(20, 12))

# 主拟合图
plt.subplot(2, 3, 1)
plt.plot(x, y_true, label='True sin(x)', linewidth=3, color='blue', alpha=0.8)
plt.scatter(x, y_noisy, label='Noisy data', s=2, alpha=0.3, color='green')
plt.plot(x, final_predictions, label=f'Prediction ({best_activation})', linewidth=2, color='red')
plt.title(f'Best Fit: Structure{best_structure} + {best_activation}\nMSE: {mse:.6f}')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)

# 训练损失
plt.subplot(2, 3, 2)
plt.plot(final_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 残差图
plt.subplot(2, 3, 3)
residuals = final_predictions.flatten() - y_true
plt.scatter(x, residuals, s=2, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.8)
plt.title('Residuals Plot')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

# 左边界详细视图
plt.subplot(2, 3, 4)
left_mask = (x >= -2*np.pi) & (x <= -3.0)
plt.plot(x[left_mask], y_true[left_mask], label='True', linewidth=3, color='blue')
plt.scatter(x[left_mask], y_noisy[left_mask], label='Noisy', s=10, alpha=0.6, color='green')
plt.plot(x[left_mask], final_predictions[left_mask], label='Prediction', linewidth=2, color='red')
plt.title('Left Boundary Detail (-2π to -3.0)')
plt.legend()
plt.grid(True, alpha=0.3)

# 右边界详细视图
plt.subplot(2, 3, 5)
right_mask = (x >= 3.0) & (x <= 2*np.pi)
plt.plot(x[right_mask], y_true[right_mask], label='True', linewidth=3, color='blue')
plt.scatter(x[right_mask], y_noisy[right_mask], label='Noisy', s=10, alpha=0.6, color='green')
plt.plot(x[right_mask], final_predictions[right_mask], label='Prediction', linewidth=2, color='red')
plt.title('Right Boundary Detail (3.0 to 2π)')
plt.legend()
plt.grid(True, alpha=0.3)

# 预测vs真实值
plt.subplot(2, 3, 6)
plt.scatter(y_true, final_predictions.flatten(), s=2, alpha=0.5)
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'r--', alpha=0.8)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.experiment_dir, 'best_fit_results.png'), dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)

# 9. 超参数搜索结果
print("\n所有超参数组合结果:")
sorted_results = sorted(results.items(), key=lambda x: x[1])
for i, (key, mse_val) in enumerate(sorted_results[:10]):  # 显示前10个最佳结果
    print(f"{i+1}. {key}: {mse_val:.6f}")

# 10. 边界区域性能分析
boundary_mask = (x <= -3.0) | (x >= 3.0)
center_mask = (x > -3.0) & (x < 3.0)

boundary_mse = np.mean((final_predictions[boundary_mask] - y_true_reshaped[boundary_mask]) ** 2) if np.any(boundary_mask) else 0
center_mse = np.mean((final_predictions[center_mask] - y_true_reshaped[center_mask]) ** 2) if np.any(center_mask) else 0

print(f"\n区域性能分析:")
print(f"中心区域 MSE (-3.0 < x < 3.0): {center_mse:.6f}")
print(f"边界区域 MSE (x <= -3.0 or x >= 3.0): {boundary_mse:.6f}")
if center_mse > 0:
    print(f"边界/中心 MSE 比率: {boundary_mse/center_mse:.3f}")
else:
    print("边界/中心 MSE 比率: N/A")

print(f"\n最佳模型配置已保存！")
print(f"最佳网络结构: {best_structure}")
print(f"最佳激活函数: {best_activation}")
print(f"最终MSE: {final_mse:.6f}")
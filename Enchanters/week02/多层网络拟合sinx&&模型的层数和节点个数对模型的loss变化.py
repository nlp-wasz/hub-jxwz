# 1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
# 增加模型的节点数目，训练时间变长，损失降低越少（反而效果变差了）
# 增加模型的层数，对于小数据集，实测，层数越少，损失降低越快
# hidden_dim_1 = 128
# hidden_dim_2 = 128
# output_dim = len(label_to_index)
# model = SimpleClassifier(vocab_size, hidden_dim_1, hidden_dim_2, output_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# 作业2：多层网络实现sinx
# 发现的多次训练该模型，会存在比较差的情况？
import numpy as np
import torch.nn as nn
import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# 生成模拟数据
X_numpy =  np.linspace(0, 10, 100)  # 100个0到10之间的随机数
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.1, 100) # 减小噪声，让曲线更明显
X = torch.from_numpy(np.expand_dims(X_numpy, axis=1)).float()
Y = torch.from_numpy(np.expand_dims(y_numpy, axis=1)).float()
print("数据生成完成。")
print("---" * 10)
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
hidden_dim = 128
input_dim = 1
output_dim = 1
model = RegressionModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    y_pred = np.sin(X)
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y_pred)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() :.4f}")
print("训练完成")



# 将模型切换到评估模式，这在训练结束后是好习惯
model.eval()

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted_tensor = model(X).numpy() # 使用训练好的模型进行预测



# 绘制结果
plt.figure(figsize=(10, 6))
# 绘制原始数据点
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# 绘制模型预测的曲线
plt.scatter(X_numpy, y_predicted_tensor, label='Model Prediction', color='red', s=10)

plt.xlabel('X')
plt.ylabel('y')
plt.title('sin x')
plt.legend()
plt.grid(True)
plt.show()



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 创建tensor
x = torch.linspace(0,2 * np.pi,1000).view(-1,1) # x定义域[0,2pi] 生产1000个点,1000 * 1 张量
y = 2 * torch.sin(2 * x + 1) + 1

class sin_model(torch.nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,output_size):
        super(sin_model,self).__init__()

        # 定义两个隐藏层
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),

            # 第4层：从 hidden_size3 到 hidden_size4
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),

            # 输出层：从 hidden_size3 到 output_size
            nn.Linear(hidden_size4, output_size)
        )

    def forward(self, x):
        return self.network(x)

model = sin_model(1,200,100,200,300,1);

# 均方差损失函数
loss_fn = torch.nn.MSELoss()

#optim是一个优化器集合体，SGD是其中一个优化器，选择优化器需要优化的变量
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

print(x.type(),x.shape)

#训练模型
for epoch in range(1000):

    # 预期结果
    y_pred = model(x)

    # 计算损失函数
    loss = loss_fn(y_pred,y)
    # 清空梯度,防止梯度累计
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()

    # 打印每一次的损失
    if((epoch + 1) % 100 == 0):
        print(f'Epoch [{epoch + 1}/{1000}] Loss {loss.item():.4f}')

# 模型切换到评估模式
model.eval()
# 不需要计算梯度
with torch.no_grad():
    y_predict = model(x).numpy()

plt.figure(figsize=(10, 7))
plt.scatter(x.data.numpy(), y.data.numpy(), label='Raw data', color='blue', alpha=0.6)
plt.plot(x.data.numpy(), y_predict,color='red', label='predicted data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

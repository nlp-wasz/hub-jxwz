import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#我预定义了一个a*sin(x)+b的函数，b的值处于(0，0.8)区间，增加数据的离散性
X_numpy=np.random.rand(100,1)*20
a=3
b=np.random.rand(100,1)*0.8
y_numpy=a*np.sin(X_numpy)+b

#训练集
X_train=torch.from_numpy(X_numpy).float()
y=torch.from_numpy(y_numpy).float()

#定义一个前馈神经网络模型：经过反复调整，发现4个隐藏层对拟合效果比较好
class ForwardNetClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(ForwardNetClassifier,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)#输入层
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_dim,64)#隐藏层
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,16)
        self.fc5=nn.Linear(16,8)
        self.fc6=nn.Linear(8,output_dim)#输出层
        
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)
        out=self.relu(out)
        out=self.fc5(out)
        out=self.relu(out)
        out=self.fc6(out)
        return out
        
model=ForwardNetClassifier(1,64,1)#初始化模型
loss_func=nn.MSELoss()#损失函数

learning_rate=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)#初始化优化器

epochs_num=4000
for epoch in range(epochs_num):
    y_train=model(X_train)

    loss=torch.mean((y_train-y)**2)#手动计算均方差

    for param in model.parameters():#手动重置每个权重参数中的梯度值
        if param.grad is not None:
            param.grad.zero_()
    
    loss.backward()#计算每个权重参数中的梯度

    #这里原本采用的是手动计算的SGD优化，但是拟合Sin函数的效果不好，所以我改用Adam优化器，直接optimizer.step()更新参数
    # with torch.no_grad():optimizer.step()更新
    #     for param in model.parameters():
    #         param-=learning_rate*param.grad      
    optimizer.step()

    if epoch%1000==0:
        print(f"损失值为：{loss.item()}")

#开始预测
model.eval()
with torch.no_grad():
    y_prediction=model(X_train)

# 矩阵排序，避免绘制时X和Y矩阵数据顺序对不上
sorted_indices = torch.argsort(X_train, dim=0).squeeze()
sorted_X = X_train[sorted_indices].numpy()
sorted_y_pred = y_prediction[sorted_indices].numpy()
    
plt.scatter(X_numpy, y_numpy,color="blue",alpha=0.5,label=f"Train Data:{a}SinX+{np.mean(b):.2f}")
plt.plot(sorted_X,sorted_y_pred,color="red",linewidth=2,label=f"Model Prediction")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


import torch
import matplotlib.pyplot as plt

# 1.创建数据
x = torch.linspace(0, 2 * torch.pi, 100)
x = x.unsqueeze(dim=1)
y = torch.sin(x) + torch.randn_like(x)


# 2.创建 全链接网络层
class TorchModel(torch.nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, out_dim):
        super(TorchModel, self).__init__()
        self.network1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1_dim, hidden2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2_dim, out_dim)
        )

        self.network2 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden1_dim, hidden2_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden2_dim, out_dim)
        )

        self.network3 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden1_dim, hidden2_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden2_dim, out_dim)
        )

        self.network4 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden1_dim, hidden2_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden2_dim, out_dim)
        )

    def forward(self, x, key):
        if key == "ReLU":
            return self.network1(x)
        elif key == "Tanh":
            return self.network2(x)
        elif key == "Sigmoid":
            return self.network3(x)
        else:
            return self.network4(x)


# 3.实例化
torchmodel1 = TorchModel(1, 10, 10, 1)
torchmodel2 = TorchModel(1, 10, 10, 1)
torchmodel3 = TorchModel(1, 10, 10, 1)
torchmodel4 = TorchModel(1, 10, 10, 1)
loss_func = torch.nn.MSELoss()
optimizer1 = torch.optim.SGD(torchmodel1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(torchmodel2.parameters(), lr=0.01)
optimizer3 = torch.optim.SGD(torchmodel3.parameters(), lr=0.01)
optimizer4 = torch.optim.SGD(torchmodel4.parameters(), lr=0.01)


# 4.模型训练
def model_loss_optimizer(torchmodule, optimizer, key):
    # 模型计算
    out_tensor = torchmodule(x, key)
    loss = loss_func(out_tensor, y)

    # 梯度清零，反向计算梯度，调参
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def model_train(torchmodel1, torchmodel2, torchmodel3, torchmodel4, optimizer1, optimizer2, optimizer3, optimizer4):
    # 模型损失值 下降对比
    lossmean_list1, lossmean_list2, lossmean_list3, lossmean_list4 = [], [], [], []
    lossmean1, lossmean2, lossmean3, lossmean4 = 0.0, 0.0, 0.0, 0.0
    count = 0

    torchmodel1.train()
    torchmodel2.train()
    torchmodel3.train()
    torchmodel4.train()
    for epoch in range(1000):
        loss1 = model_loss_optimizer(torchmodel1, optimizer1, "ReLU")
        loss2 = model_loss_optimizer(torchmodel2, optimizer2, "Tanh")
        loss3 = model_loss_optimizer(torchmodel3, optimizer3, "Sigmoid")
        loss4 = model_loss_optimizer(torchmodel4, optimizer4, "LeakyReLU")

        print(f"激活函数：ReLU，循环第{epoch}次，loss：{loss1}")
        print(f"激活函数：Tanh，循环第{epoch}次，loss：{loss2}")
        print(f"激活函数：Sigmoid，循环第{epoch}次，loss：{loss3}")
        print(f"激活函数：LeakyReLU，循环第{epoch}次，loss：{loss4}")
        print("-" * 100)

        # 用来对比模型差异
        # 模型损失值 下降对比
        lossmean1 += loss1.item()
        lossmean2 += loss2.item()
        lossmean3 += loss3.item()
        lossmean4 += loss4.item()
        count += 1
        if epoch % 10 == 0:
            lossmean1 /= count
            lossmean2 /= count
            lossmean3 /= count
            lossmean4 /= count
            lossmean_list1.append(lossmean1)
            lossmean_list2.append(lossmean2)
            lossmean_list3.append(lossmean3)
            lossmean_list4.append(lossmean4)

            # 清零
            lossmean1, lossmean2, lossmean3, lossmean4 = 0.0, 0.0, 0.0, 0.0
            count = 0

    # 模型损失值 下降对比
    # 画图 展示模型之间的差异
    plt.figure(figsize=(12, 4))
    plt.plot(lossmean_list1, label=f"ReLU", color="red")
    plt.plot(lossmean_list2, label=f"Tanh", color="orange")
    plt.plot(lossmean_list3, label=f"Sigmoid", color="yellow")
    plt.plot(lossmean_list4, label=f"LeakyReLU", color="green")
    plt.xlabel("epoch")
    plt.ylabel("lossmean")
    plt.legend()
    plt.show()


model_train(torchmodel1, torchmodel2, torchmodel3, torchmodel4, optimizer1, optimizer2, optimizer3, optimizer4)

# 5.图表展示
torchmodel1.eval()
torchmodel2.eval()
torchmodel3.eval()
torchmodel4.eval()
with torch.no_grad():
    pred_y1 = torchmodel1(x, "ReLU")
    pred_y2 = torchmodel2(x, "Tanh")
    pred_y3 = torchmodel3(x, "Sigmoid")
    pred_y4 = torchmodel4(x, "LeakyReLU")

# 模型训练 结果对比
plt.figure(figsize=(12, 4))
plt.scatter(x, y, label="bdfore")
plt.plot(x, pred_y1, label="ReLU", color="red")
plt.plot(x, pred_y2, label="Tanh", color="orange")
plt.plot(x, pred_y3, label="Sigmoid", color="yellow")
plt.plot(x, pred_y4, label="LeakyReLU", color="green")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# 1. 生成数据（归一化到 [0, 2π]）
X_numpy = np.random.rand(500, 1) * 2 * np.pi
y_numpy = np.sin(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 2. 定义更复杂的模型
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        y_pred = self.net(x)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


# 3. 训练
def main():
    model = TorchModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(200):
        model.train()
        loss = model(X, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 4. 预测并绘图
    with torch.no_grad():
        y_pred = model(X).numpy()

    plt.scatter(X_numpy, y_numpy, label="Real", s=5)
    plt.scatter(X_numpy, y_pred, label="Pred", s=5, c="red")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
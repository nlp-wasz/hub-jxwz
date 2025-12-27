import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1) 造数据：sin(x) + 噪声
np.random.seed(42)
torch.manual_seed(42)

N = 512
x_np = np.linspace(-2*np.pi, 2*np.pi, N).reshape(-1, 1)
noise = 0.10 * np.random.randn(N, 1)
y_np = np.sin(x_np) + noise

x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

# 2) 定义多层网络（MLP）
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 3) 训练
epochs = 1000
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.6f}")

# 4) 可视化
model.eval()
with torch.no_grad():
    grid = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1).astype(np.float32)
    pred = model(torch.from_numpy(grid)).numpy()

plt.figure(figsize=(9, 4.5))
plt.scatter(x_np, y_np, s=10, alpha=0.5, label='Train')
plt.plot(grid, np.sin(grid), 'g--', linewidth=2, label='True sin(x)')
plt.plot(grid, pred, 'r', linewidth=2, label='MLP fit')
plt.title('Fit sin(x) with MLP (Tanh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('sin_fit.png')
plt.show()
# LSTM (Long Short-Term Memory)
RNN 的改进版，旨在解决 RNN 在处理⻓序列时出现的梯度消失和梯度爆炸问题。LSTM 通过引入门控机制（输⼊⻔、遗忘⻔、输出⻔）和细胞状态来解决传统RNN的梯度消失问题，能够更好地捕捉长期依赖关系。

```python
class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
```

## LSTM 基本计算原理
LSTM 的三个门控
1. 遗忘门（Forget Gate）
决定从细胞状态中丢弃哪些信息
2. 输入门（Input Gate）
决定哪些新信息要存储到细胞状态中
3. 输出门（Output Gate）
决定基于细胞状态输出什么信息

### LSTM 计算公式

### 遗忘门（Forget Gate）
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

### 输入门（Input Gate）
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

### 细胞状态更新
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
### 输出门（Output Gate）
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前时间步的输入 |
| $h_{t-1}$ | 上一个时间步的隐藏状态 |
| $C_{t-1}$ | 上一个时间步的细胞状态 |
| $W_f, W_i, W_o, W_C$ | 权重矩阵 |
| $b_f, b_i, b_o, b_C$ | 偏置向量 |
| $\sigma$ | Sigmoid激活函数（输出0-1） |
| $\tanh$ | 双曲正切激活函数（输出-1到1） |
| $\odot$ | 逐元素乘法（Hadamard积） |

### 计算流程示意图

```
输入: xₜ, hₜ₋₁, Cₜ₋₁
    ↓
遗忘门: fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)  // 决定遗忘什么
    ↓
输入门: iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)  // 决定更新什么
    ↓
候选值: C̃ₜ = tanh(W_C·[hₜ₋₁, xₜ] + b_C) // 新信息的候选
    ↓
细胞状态: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ     // 更新细胞状态
    ↓
输出门: oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)  // 决定输出什么
    ↓
隐藏状态: hₜ = oₜ ⊙ tanh(Cₜ)          // 计算新隐藏状态

## PyTorch 中的 LSTM 实现

```python
import torch
import torch.nn as nn

# 定义LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# 输入数据
x = torch.randn(1, 5, 10)  # (batch_size, seq_len, input_size)
h0 = torch.zeros(1, 1, 20)  # 初始隐藏状态
c0 = torch.zeros(1, 1, 20)  # 初始细胞状态

# 前向传播
output, (hn, cn) = lstm(x, (h0, c0))
# output: 所有时间步的隐藏状态
# hn: 最终隐藏状态
# cn: 最终细胞状态
```

## LSTM 的优势

1. **解决梯度消失**：细胞状态的线性循环连接使得梯度可以长时间流动
2. **选择性记忆**：门控机制可以学习何时记住、何时遗忘信息
3. **长期依赖**：能够捕捉长距离的时间依赖关系

## 与普通RNN的对比

| 特性 | 普通RNN | LSTM |
|------|---------|------|
| 记忆机制 | 简单隐藏状态 | 细胞状态 + 门控 |
| 梯度问题 | 容易梯度消失 | 缓解梯度消失 |
| 长期依赖 | 处理能力有限 | 强长期记忆能力 |
| 参数数量 | 较少 | 较多（4倍于RNN） |
LSTM通过这些精妙的门控机制，实现了对信息的精细化控制，使其在各种序列建模任务中表现出色。

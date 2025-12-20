# GRU
GRU是对LSTM的巧妙的简化：
合并冗余门控：将功能相关的遗忘门和输入门合并为更新门
重新设计输出控制：用重置门替代输出门，更直接地控制历史信息使用
减少参数数量：从3个门减到2个门，提高计算效率
保持表达能力：尽管简化，但在大多数任务上性能相当
这种设计体现了深度学习中的一个重要原则：在**保持模型表达能力的同时，尽可能简化结构**。GRU的成功也启发了后来更多高效的网络结构设计。

```python
class torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
```

# 1. GRU 计算原理详解

## GRU 的核心思想

GRU通过**两个门控**（重置门和更新门）来平衡新信息和历史信息，比LSTM更简洁但同样有效。

## GRU 的两个门控

### 1. 重置门（Reset Gate）
决定如何将新输入与 previous 隐藏状态结合

### 2. 更新门（Update Gate）
决定保留多少历史信息，接受多少新信息

## GRU 计算公式（Markdown形式）

### 更新门（Update Gate）
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

### 重置门（Reset Gate）
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

### 候选隐藏状态
$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

### 最终隐藏状态
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

## 公式符号说明

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前时间步的输入 |
| $h_{t-1}$ | 上一个时间步的隐藏状态 |
| $z_t$ | 更新门输出（0-1） |
| $r_t$ | 重置门输出（0-1） |
| $\tilde{h}_t$ | 候选隐藏状态 |
| $W_z, W_r, W$ | 权重矩阵 |
| $b_z, b_r, b$ | 偏置向量 |
| $\sigma$ | Sigmoid激活函数 |
| $\tanh$ | 双曲正切激活函数 |
| $\odot$ | 逐元素乘法 |

## 计算流程示意图

```text
输入: xₜ, hₜ₋₁
    ↓
更新门: zₜ = σ(W_z·[hₜ₋₁, xₜ] + b_z)
    ↓
重置门: rₜ = σ(W_r·[hₜ₋₁, xₜ] + b_r)  
    ↓
候选状态: h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ] + b)
    ↓
最终状态: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

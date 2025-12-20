# RNN
最基本的循环神经⽹络。它通过⼀个隐藏状态（Hidden State）来捕获序列历史信息。在处理序列的每个时间步时，RNN 会将当前输
⼊和上⼀个时间步的隐藏状态作为输⼊，并⽣成新的隐藏状态和输出。

```python
class torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
```

## RNN基本计算原理
RNN的核心思想是循环，它通过循环神经单元来处理序列数据，其基本计算公式为

$$ h_t=tanh(x_t \cdot W_{ih}^T+b_{ih}+h_{t-1} \cdot W_{hh}^T +b_{hh})$$
其中，$h_t$为当前时间步的隐藏状态；$h_{t-1}$为上一个时间步的隐藏状态，或时间0时刻的隐藏状态；
$x_t$为当前时间步的输入；$W_{ih}$为输入到隐藏层的权重矩阵；$W_{hh}$为隐藏层到隐藏层的权重矩阵；
$b_{ih}$,$b_{hh}$对应的偏置项；tanh为激活函数（也可以是其他函数，如ReLu）

## 计算过程
时间步 t 的计算：
     输入 xₜ
        ↓
     Wᵢₕ·xₜ + bᵢₕ
        ↓
        + ←--- Wₕₕ·hₜ₋₁ + bₕₕ
        ↓
     tanh(·)
        ↓
     输出 hₜ

- 将当前输入信息和历史信息相加
- 通过 tanh 激活函数进行非线性变换
- 得到新的隐藏状态 hₜ

## RNN的特点
参数共享：所有时间步使用相同的权重参数 Wᵢₕ 和 Wₕₕ
记忆功能：隐藏状态 hₜ 包含了之前所有时间步的信息
序列处理：能够处理可变长度的序列数据

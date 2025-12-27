# text_classifier.py中模型解释
## 网络结构图解
```markdown
graph LR
    A[输入层<br>vocab_size] --> B[全连接层1<br>hidden_dim]
    B --> C[ReLU激活]
    C --> D[Dropout1<br>p=0.5]
    D --> E[全连接层2<br>hidden_dim//2]
    E --> F[ReLU激活]
    F --> G[Dropout2<br>p=0.25]
    G --> H[输出层<br>output_dim]
```

## 各层详细说明
### 1. 输入层
- 输入形状：[batch_size, vocab_size]
- 数据形式：词袋向量（BoW），每个元素表示对应字符的出现次数
- 示例：若词表有2823个字符，则每个文本表示为2823维的稀疏向量

### 2. 第一全连接层
```python
self.fc1 = nn.Linear(vocab_size, hidden_dim)
```
作用：将高维稀疏向量压缩到低维稠密表示
参数计算：
权重矩阵形状：`[vocab_size, hidden_dim]`
偏置向量形状：`[hidden_dim]`
典型值：若 vocab_size=2823, hidden_dim=256，则参数量为 2823×256 + 256 ≈ 723k

### 3. 第一个Dropout层
```python
self.dropout1 = nn.Dropout(Config.dropout_prob)  # p=0.5
```
作用：随机丢弃50%神经元输出，防止过拟合
仅在训练时激活：通过 model.train()/model.eval() 自动切换

### 4. 第二个全连接层fc2
```python
self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # 256→128
```
作用： 逐步降低维度，提取高阶特征
参数减半：通过//2实现维度压缩，减少计算量

### 5. 第二个Dropout层
```python
self.dropout2 = nn.Dropout(Config.dropout_prob / 2)  # p=0.25
```
更小的丢弃率：靠近输出层时减少随机性，保留更多信息

### 6. 输出层fc3
```python
self.fc3 = nn.Linear(hidden_dim // 2, output_dim)  # 128→类别数
```
无激活函数：直接输出logits（配合CrossEntropyLoss使用）
输出解释：每个值对应类别的原始分数

## 前向传播流程
```python
def forward(self, x):
    x = F.relu(self.fc1(x))     # [B, vocab_size] → [B, 256]
    x = self.dropout1(x)        # 训练时随机置零50%元素
    x = F.relu(self.fc2(x))     # [B, 256] → [B, 128]
    x = self.dropout2(x)        # 训练时随机置零25%元素
    return self.fc3(x)          # [B, 128] → [B, output_dim]
```

## 关键设计思想
### 1. 维度递减结构
vocab_size → hidden_dim → hidden_dim//2 → output_dim
逐步压缩信息，避免直接从高维到低维的剧烈维度变化

### 2. 渐进式Dropout
第一层Dropout率较高（0.5），后续降低（0.25）
符合"底层学习通用特征，高层学习具体特征"的认知

### 3. 激活函数选择
使用ReLU而非Sigmoid/Tanh：
缓解梯度消失问题
计算效率更高
更适合深层网络

### 4. 与输入数据的适配
输入为词袋向量（非负计数），ReLU能保持这种非负特性
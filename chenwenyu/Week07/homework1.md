DeepWalk原理
DeepWalk的核心思想是将图结构数据转化为自然语言处理中的文本序列，从而可以借用Word2Vec等NLP技术来学习节点表示。

主要步骤：
随机游走生成：

从每个节点出发，进行多次随机游走

每次游走固定长度（walk_length=20）

游走路径类似于自然语言中的句子

节点序列化：

将图结构转化为节点序列的集合

例如：["Researcher_1", "Researcher_3", "Artist_2", ...]

类比自然语言：

节点 ⇨ 单词

随机游走序列 ⇨ 句子

整个图的游走集合 ⇨ 语料库

Word2Vec的作用
在DeepWalk中，Word2Vec负责将节点序列学习为低维向量表示：

具体功能：
向量化表示：

python
# 将每个节点映射为64维向量
model = Word2Vec(walks, vector_size=64, window=5, sg=1)
Skip-gram模型：

sg=1 表示使用Skip-gram算法

目标：给定中心节点，预测其上下文节点

使拓扑相近的节点在向量空间中也相近

窗口机制：

window_size=5 定义上下文窗口大小

在游走序列中，距离不超过5的节点被认为是"上下文相关"
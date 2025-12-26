## GTP4Rec的整体思路
用 GPT 把“推荐”转化为“生成用户兴趣搜索词 → 搜索引擎检索物品”的问题
也就是 “生成”+“检索”

## 整体系统架构
1. 论文中的推荐系统可以拆成5个工程模块：
用户历史 → Prompt 构造
        ↓
   GPT（生成多个 Query）
        ↓
   搜索引擎（BM25 / 向量检索）
        ↓
   合并 & 排序
        ↓
   推荐结果

2. 实施步骤
Step 1. 准备数据
用户历史行为序列，即一串item titles（不是item ID）
GPT4Rec 完全不用 item ID，而是直接用文本（标题）
这就是它能处理冷启动、动态商品池的核心原因。

Step 2. 构造提示词（把推荐变成语言任务）
论文中固定使用如下模板
```text
Previously, the customer has bought:
<Item Title 1>. <Item Title 2>. ... <Item Title T>.

In the future, the customer wants to buy 
```
Step 3. 微调GPT（Query Generator）
GPT的任务不是“直接输出推荐商品”，而是生成“用户可能搜索的查询语句（Query）”
训练方式：
论文采用 Next-Item Title Prediction：
输入：用户前 T-1 个 item titles + Prompt
输出（监督信号）：第 T 个 item 的 title
训练目标：最小 P( item_T_title | history_titles + prompt )

Step 4. 多Query生成
Step 5. 检索模块
生成的Query会送入搜索引擎，论文使用BM25，本质是文本相关性打分。
Step 6. 多Query结果融合
总推荐数=K，Query数=m
对第一个Query取K/m个结果，对第二个Query，再取K/m个结果...
保证相关性和多样性。

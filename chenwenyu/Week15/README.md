[TOC]

## task1
### 任务描述
阅读【08-financial-report-generator】中包含的方案，总结下其中使用了哪些外部数据源？
### 外部数据源
- Wind
- Akshare
- 沪深300上证指数
- 东方财富
- 同花顺
- 金十数据
- 上交所、深交所、港交所、期货交易所
- gov.cn、国家统计局、人民银行
- 美联储官网、世银

## task2
### 任务描述
对NLP_大模型知识点和面试题_V0_1.pdf进行解析，对文档进行RAG问答。

### 实现方案
本地部署mineru，将pdf文件转换为md文件后，进行清理并保存为文件，NLP_面试题_fixed.md
基于该文件建立RAG

- 快速通过正则表达式实现，代码文件：questions_rag_re.py
$ python questions_rag_re.py
面试题RAG系统已启动！
输入关键词搜索相关问题，输入'quit'退出
--------------------------------------------------

请输入搜索关键词: 词向量

找到 8 个相关问题:
--------------------------------------------------
1. [原题40] 如何利⽤词向量（如 Word2Vec 或 GloVe 的嵌⼊）计算两个词或句⼦之间的语义相似度？请列出具体的计算步骤。...
   匹配度: 3, 关键词: 词向量

2. [原题39] 随着BERT等预训练语⾔模型的出现，传统的静态词向量（如Word2Vec）在应⽤中的地位发⽣了哪些变化？它们各⾃更适⽤于哪些场景？...
   匹配度: 1, 关键词: 词向量

3. [原题41] 除了余弦相似度，还有哪些⽅法或指标可以衡量词向量空间中的语义关系？...
   匹配度: 1, 关键词: 词向量

4. [原题42] 词向量可以捕捉哪些类型的语义关系？请举例说明。它通常在哪些关系上表现不佳？...
   匹配度: 1, 关键词: 词向量

5. [原题43] 在处理同义词和多义词时，静态词向量（如Word2Vec）存在什么局限性？后续的模型（如ELMo,BERT）是如何试图解决这些问题的？...
   匹配度: 1, 关键词: 词向量

6. [原题44] 请列举并⽐较⾄少三种基于词向量构建句⼦向量的经典⽅法（例如，简单平均、加权平均、 使⽤RNN/LSTM编码等）。...
   匹配度: 1, 关键词: 词向量

7. [原题47] 在什么情况下，对词向量取平均作为句⼦向量是⼀个有效或⽆效的策略？请从任务类型和⽂ 本特点⻆度分析。...
   匹配度: 1, 关键词: 词向量

8. [原题195] RoPE的核⼼思想是什么？它是如何将绝对位置信息通过旋转矩阵融⼊词向量的？...
   匹配度: 1, 关键词: 词向量


请输入搜索关键词: Transformer

找到 10 个相关问题:
--------------------------------------------------
1. [原题68] 为什么 Transformer 中需要残差连接？...
   匹配度: 3, 关键词: transformer

2. [原题69] Transformer 中的 LayerNorm 跟 ResNet 中的 BatchNorm 有什么区别，为什么 LLaMA-3 换⽤了 RMSNorm？...
   匹配度: 3, 关键词: transformer

3. [原题76] 从统计学⻆度看，Transformer 输出层假设词元符合什么分布？...
   匹配度: 3, 关键词: transformer

4. [原题94] 相对位置编码（如 T5, Transformer-XL, RoPE 中使⽤的⽅法）与绝对位置编码相⽐，主要优势是什么？...
   匹配度: 3, 关键词: transformer

5. [原题100] 什么是“Encoder-Decoder”架构（或称“Seq2Seq with Transformer”）？请列举⼀个典型模型，并说明它在什么任务上表现出⾊。...
   匹配度: 3, 关键词: transformer

6. [原题31] Transformer中的编码器和解码器有什么区别，只有编码器或者只有解码器的模型是否有⽤？...
   匹配度: 1, 关键词: transformer

7. [原题32] GPT跟原始Transformer论⽂的模型架构有什么区别？...
   匹配度: 1, 关键词: transformer

8. [原题34] 为什么说Transformer的⾃注意⼒机制相对于早期RNN中的注意⼒机制是⼀个显著的进步？...
   匹配度: 1, 关键词: transformer

9. [原题48] 请详细描述Transformer模型的整体架构，包括编码器和解码器的组成、数据流，并解释其为何能并⾏处理序列。...
   匹配度: 1, 关键词: transformer

10. [原题51] 为什么Transformer需要“位置编码”？请解释原始Transformer中使⽤正弦/余弦函数进⾏位置编码的原理。...
   匹配度: 1, 关键词: transformer


请输入搜索关键词: quit
再见！祝面试成功！

- 通过语义以及正则表达式混合实现，代码文件：questions_rag_semantic.py

$ python questions_rag_semantic.py --md_file ./NLP_面试题_fixed.md  --interactive
加载模型: /root/autodl-tmp/models/google-bert/bert-base-chinese...
生成新的嵌入向量...
生成嵌入向量...
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.48s/it]
已处理: 32/275
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.21s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.97s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.39s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.03s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.19s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.40s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.99s/it]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.89s/it]
嵌入向量生成完成，形状: (275, 768)
构建FAISS索引...
FAISS索引构建完成，包含 275 个向量
缓存已保存: 9f35e6b78b79b8c5
============================================================
语义RAG系统 - 面试题搜索
============================================================
命令:
  /s <关键词>    - 语义搜索
  /h <关键词>    - 混合搜索
  /k <关键词>    - 关键词搜索
  /sim <问题ID>  - 查找相似问题
  /cluster       - 聚类分析
  /quit          - 退出
============================================================

请输入命令: /s 大模型

语义搜索: '大模型'
搜索结果 (共 10 个):

1. [原题102] 如何基于表⽰型模型⽣成的嵌⼊向量实现⽂本分类？
2. [原题118] 如何保证模型的输出⼀定是合法的JSON格式？（提⽰：限制采样）
3. [原题64] 训练时如何防⽌模型看到未来的词元？
4. [原题234] 如何训练⼀个有效的重排模型？其训练数据通常如何构造？
5. [原题32] GPT跟原始Transformer论⽂的模型架构有什么区别？
6. [原题63] ⼤模型怎么知道它的输出该结束了？
7. [原题28] 什么是⼤型语⾔模型？请从其设计⽬标、核⼼能⼒和典型代表模型三个⽅⾯进⾏阐述。
8. [原题68] 为什么 Transformer 中需要残差连接？
9. [原题110] 对于指定的⼤模型，如何通过提⽰词减少其幻觉？
10. [原题156] “涌现能⼒”通常在模型规模超过某个阈值时出现。请解释这种现象背后的可能原因（可从缩放定律、模型容量等⻆度分析）。


请输入命令: /sim 102

查找与问题 102 相似的问题:
搜索结果 (共 5 个):

1. [原题103] 使⽤嵌⼊向量实现分类和使⽤⽣成式模型直接分类的⽅法相⽐，有什么优缺点？
2. [原题104] 如果没有标注数据，如何基于嵌⼊模型实现⽂本分类？如何优化标签描述来提⾼零样本分类的准确率？
3. [原题6] GloVe模型是如何将全局统计信息（如共现矩阵）与局部上下⽂窗⼝⽅法结合起来的？
4. [原题98] 在 BERT 之后的预训练模型（如 RoBERTa, ALBERT, DeBERTa）中，这些嵌⼊类型或输⼊表⽰发⽣了哪些变化或优化？
5. [原题121] 掩码语⾔建模与BERT的掩蔽策略相⽐有何不同？这种预训练⽅式如何帮助模型在下游的⽂本分类任务中获得更好的性能？


请输入命令: /quit
再见！祝面试成功！

## Task3 DeepResearch Agent
### 任务描述
将 09_DeepResearch.py 改为不同章节同时生成，并且加入 方式/react 的机制，大模型判断这个这个章节的生成效果，有反馈建议，逐步生成。

### 主要改动
增加函数process_section_with_feedback，主要实现
- 并行进行各章节的草稿
- 增加feedback Agent，让大模型评估章节的质量，提供反馈，对章节进行改进。

将报告保存为md文件

```python
async def process_section_with_feedback(section: Dict[str, Any], max_retries: int = 2) -> str:

...
```

((dl_venv) ) root@...homework $ python 09_DeepResearch.py

--- Deep Research for: Agentic AI在软件开发中的最新应用和挑战 ---

Step 1: 进行初步检索...
-> [Jina Search] 正在搜索: Agentic AI在软件开发中的最新应用和挑战...
[{"title": "Agentic AI基础设施实践经验系列（七）：可观测性在Agent应用的 ...", "url": "https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-7/", "snippet": ""}, {"title": "Agentic AI 简易指南 - Mendix", "url": "https://www.mendix.com/zh-CN/%E6%96%B0%E9%97%BB/%E4%BB%A3%E7%90%86%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E6%8C%87%E5%8D%97/", "snippet": ""}, {"title": "从AI Agent到Agentic AI，探秘人工智能从“工具”到“行动者”的进阶之路", "url": "https://zhuanlan.zhihu.com/p/1921640177593475588", "snippet": ""}, {"title": "AI Agent与Agentic AI：原理、应用、挑战与未来展望_人工智能", "url": "https://deepseek.csdn.net/68639397a6db534ba2b536f5.html", "snippet": ""}, {"title": "什么是代理式AI？ - Red Hat", "url": "https://www.redhat.com/zh-cn/topics/ai/what-is-agentic-ai", "snippet": ""}, {"title": "实践指南：探索Agentic AI 的复杂性 - IBM", "url": "https://www.ibm.com/cn-zh/think/insights/navigating-the-complexities-of-agentic-ai", "snippet": ""}, {"title": "Agentic AI基础设施实践经验系列（一）：Agent应用开发与 ... - AWS", "url": "https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/", "snippet": ""}, {"title": "什么是agentic AI？ | Oracle 中国", "url": "https://www.oracle.com/cn/artificial-intelligence/agentic-ai/", "snippet": ""}, {"title": "甚麼是代理式人工智能（Agentic AI）？ | 趨勢科技 - Trend Micro", "url": "https://www.trendmicro.com/zh_hk/what-is/ai/agentic-ai.html", "snippet": ""}, {"title": "AI Agent在多领域应用：面临哪些挑战与机遇？", "url": "https://docs.feishu.cn/v/wiki/Bt1bw7ns7itObLk8LTTcTBh8njf/ah", "snippet": ""}]

Step 2: 基于初步结果生成研究大纲...
RunResult:
- Last agent: Agent(name="Deep Research Orchestrator", ...)
- Final output (str):
    {
        "title": "Agentic AI在软件开发中的最新应用和挑战",
        "sections": [
            {"section_title": "引言与背景", "search_keywords": "Agentic AI 历史, 当前发展状况"},
            {"section_title": "Agentic AI的核心概念与技术基础", "search_keywords": "Agentic AI 关键概念, 技术原理"},
            {"section_title": "Agentic AI的软件开发生态系统", "search_keywords": "Agentic AI 生态系统, 开发工具"},
            {"section_title": "Agentic AI在软件开发中的具体应用场景", "search_keywords": "Agentic AI 应用案例, 软件项目"},
            {"section_title": "Agentic AI面临的挑战及解决方案", "search_keywords": "Agentic AI 挑战, 解决方案"},
            {"section_title": "未来趋势与展望", "search_keywords": "Agentic AI 未来发展, 技术趋势"}
        ]
    }
- 1 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResult` for more details)
报告标题: Agentic AI在软件开发中的最新应用和挑战
规划了 5 个章节。

--- Step 3: 带有反馈机制的并行章节生成 ---

[处理章节] 引言与背景
  第1轮生成...

[处理章节] Agentic AI的核心概念与技术基础
-> [Jina Search] 正在搜索: 引言与背景 搜索关键词: Agentic AI 历史, 当前发展状况...
  第1轮生成...
-> [Jina Search] 正在搜索: Agentic AI的核心概念与技术基础 搜索关键词: Agentic AI 关键概念, 技术原理...

[处理章节] Agentic AI的软件开发生态系统
  第1轮生成...
-> [Jina Search] 正在搜索: Agentic AI的软件开发生态系统 搜索关键词: Agentic AI 生态系统, 开发工具...

[处理章节] Agentic AI在软件开发中的具体应用场景
  第1轮生成...
-> [Jina Search] 正在搜索: Agentic AI在软件开发中的具体应用场景 搜索关键词: Agentic AI 应用案例, 软件...

[处理章节] Agentic AI面临的挑战及解决方案
  第1轮生成...
-> [Jina Search] 正在搜索: Agentic AI面临的挑战及解决方案 搜索关键词: Agentic AI 挑战, 解决方案...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1951222493097498243...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1918252389946861046...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentive-ai-...
-> [Jina Crawl] 正在抓取: https://www.redhat.com/zh-cn/topics/ai/what-is-age...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://blog.csdn.net/qq_46094651/article/details/...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1968814143378290608...
-> [Jina Crawl] 正在抓取: https://modelers.csdn.net/6913e8415511483559e85025...
Error during Jina Crawl for https://zhuanlan.zhihu.com/p/1918252389946861046: HTTPSConnectionPool(host='r.jina.ai', port=443): Read timed out. (read timeout=20)
  草稿生成完成
  草稿生成完成
  草稿生成完成
  质量评分: 3/5
  继续优化...
  第2轮生成...
-> [Jina Search] 正在搜索: Agentic AI的软件开发生态系统 搜索关键词: Agentic AI 生态系统, 开发工具...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentive-ai-...
-> [Jina Crawl] 正在抓取: https://www.redhat.com/zh-cn/topics/ai/what-is-age...
  草稿生成完成
  质量评分: 3/5
  继续优化...
  第2轮生成...
-> [Jina Search] 正在搜索: 引言与背景 搜索关键词: Agentic AI 历史, 当前发展状况...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1968814143378290608...
-> [Jina Crawl] 正在抓取: https://modelers.csdn.net/6913e8415511483559e85025...
  质量评分: 3/5
  继续优化...
  第2轮生成...
-> [Jina Search] 正在搜索: Agentic AI的核心概念与技术基础 搜索关键词: Agentic AI 关键概念, 技术原理...
  质量评分: 3/5
  继续优化...
  第2轮生成...
-> [Jina Search] 正在搜索: Agentic AI面临的挑战及解决方案 搜索关键词: Agentic AI 挑战, 解决方案...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1918252389946861046...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1951222493097498243...
  草稿生成完成
  草稿生成完成
  质量评分: 3/5
  继续优化...
  第2轮生成...
-> [Jina Search] 正在搜索: Agentic AI在软件开发中的具体应用场景 搜索关键词: Agentic AI 应用案例, 软件...
  质量评分: 3/5
  继续优化...
  第3轮生成...
-> [Jina Search] 正在搜索: Agentic AI的软件开发生态系统 搜索关键词: Agentic AI 生态系统, 开发工具...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentive-ai-...
-> [Jina Crawl] 正在抓取: https://www.redhat.com/zh-cn/topics/ai/what-is-age...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://blog.csdn.net/qq_46094651/article/details/...
  草稿生成完成
  质量评分: 4/5
  质量达标，结束优化
  ✓ 章节完成 (最终评分: 4/5)
  草稿生成完成
  ✓ 章节完成 (最终评分: 3/5)
  草稿生成完成
  草稿生成完成
  草稿生成完成
  质量评分: 3/5
  继续优化...
  第3轮生成...
-> [Jina Search] 正在搜索: Agentic AI在软件开发中的具体应用场景 搜索关键词: Agentic AI 应用案例, 软件...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://blog.csdn.net/qq_46094651/article/details/...
  质量评分: 4/5
  质量达标，结束优化
  ✓ 章节完成 (最终评分: 4/5)
  质量评分: 3/5
  继续优化...
  第3轮生成...
-> [Jina Search] 正在搜索: Agentic AI的核心概念与技术基础 搜索关键词: Agentic AI 关键概念, 技术原理...
-> [Jina Crawl] 正在抓取: https://aws.amazon.com/cn/blogs/china/agentic-ai-i...
-> [Jina Crawl] 正在抓取: https://zhuanlan.zhihu.com/p/1951222493097498243...
Error during Jina Crawl for https://zhuanlan.zhihu.com/p/1951222493097498243: HTTPSConnectionPool(host='r.jina.ai', port=443): Read timed out. (read timeout=20)
  草稿生成完成
  ✓ 章节完成 (最终评分: 3/5)
  草稿生成完成
  ✓ 章节完成 (最终评分: 3/5)

Step 4: 整合最终研究报告...
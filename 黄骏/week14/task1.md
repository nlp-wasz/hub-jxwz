# 思路一 RAG + Qwen Thinking 模型（LLM主导计算）

步骤一：pdf公式解析，可以使用qwen-vl或者mineru解析pdf为markdown格式，并区分Latex公式，参数解释，公式应用背景 三部分。这里应该是对公式应用背景部分和公式部分分别进行embedding

步骤二：RAG检索与排序，将用户提问进行embedding，分别与公式和公式背景进行相似度计算，然后进行rerank，选取前top1-8，然后合并去重。

步骤三：使用qwen-3 thinking模型，输入用户提问和公式latex，参数解释，生成对应的代码，进行执行。返回结果；

或者可以直接推理得到答案。

## 待选方案一

1. pdf解析，使用了dots.ocr，解析为markdown，每个markdown整体进行embedding，
2. RAG检索，对用户提问进行embedding，然后选取TOP-8文档全部加入提示词当中
3. 利用提示词设计，让大模型自己筛选做合适的公式进行计算

## 待选方案二

1. pdf解析，使用pypdf2进行解析
2. 文本匹配，将问题和文档进行匹配
3. 利用提示词设计，让大模型根据文档进行计算

## 待选方案三

1. pdf解析，使用fitz进行解析，每个markdown进行embedding
2. RAG检索，对用户提问进行embedding，选择TOP-8全部加入提示词中
3. 大模型回答，将用户提问分为三类，数值预测、公式预测、影响预测。如果是数值预测，则将对应的公式使用sympy进行数值计算，得出结果
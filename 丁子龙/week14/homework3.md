# 结合rag tool筛选 加 mcp tool执行完成 整个问答

## 执行步骤
1. 文档解析（pdf-> raw markdown）
2. 文档内容划分。（公式背景，公式与公式参数）
3. embedding和mcp代码生成 --> 将embedding,原文本，公式，mcp代码，mcp函数名保存到数据库中，mcp代码到mcp server代码文件中
4. RAG检索，将用户提问进行embedding，执行混合搜索，然后执行rerank，选取TOP-8
5. 将top-8mcp工具名加入白名单中，设计提示词模板，让大模型自动筛选并执行mcp，获取结果
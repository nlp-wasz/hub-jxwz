
# 作业3: 阅读政企问答项目的代码，自己描述RAG的实现流程，写为文档。

"""
1. 文档入库流程
1.1 API入口
路由：POST /v1/document
主要逻辑：
校验知识库是否存在（查KnowledgeDatabase）。
新建KnowledgeDocument记录，生成document_id。
保存上传文件到本地。
更新KnowledgeDocument.file_path。
后台任务：调用RAG().extract_content进行文档内容解析与入库。
1.2 文档内容解析（RAG.extract_content）
判断文件类型（如PDF），调用_extract_pdf_content。
逐页提取文本，抽取摘要。
每页内容切分为chunk，分别生成embedding向量。
每个chunk及其向量、元数据写入ES的chunk_info索引。
文档元数据写入document_meta索引。
2. 检索与问答流程
2.1 用户提问
路由：POST /chat
主要逻辑：
调用RAG().chat_with_rag(knowledge_id, message)。
2.2 检索与生成（RAG.chat_with_rag）
取出用户问题（query）。
调用query_document(query, knowledge_id)检索相关chunk。
2.2.1 检索逻辑（RAG.query_document）
全文检索：在ES中用BM25对chunk内容检索。

向量检索：对query编码，做向量相似度检索。

融合排序：用RRF算法融合两种检索结果，选出topN候选chunk。

重排序（可选）：用重排序模型对候选chunk与query做相关性打分，再排序。

返回最相关的chunk内容列表。

2.2.2 Prompt构造与生成
将检索到的chunk内容拼接，填入BASIC_QA_TEMPLATE模板。
调用RAG.chat，用大模型API生成答案。
3. 其他API说明
/v1/embedding：调用RAG().get_embedding，返回文本embedding向量。
/v1/rerank：调用RAG().get_rank，返回文本对的相关性分数。
/v1/knowledge_base、/v1/document等：分别用于知识库和文档的增删查。


"""
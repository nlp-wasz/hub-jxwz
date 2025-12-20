# 1.deepseek API申请

![2025-09-11_09-49.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/a7e98e51-607a-4bb7-9c13-cc78080694be.jpeg)

# 2.ollama安装和qwen3:0.6b部署

![2025-09-11_09-52.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/1bca7515-9a78-408c-ab66-316a21ca5d2f.jpeg)

# 3.政企问答项目RAG实现流程

1. 首先查看需求，一般而言增删查改、根据特定主题回答问题、多轮对话等功能。
2. 设置接口至少需要增删查改和对话的接口，针对这一个项目需要文档的增删查改，知识问答的增删查改，还有对话，至少九个接口。
3. 针对于知识问答的增删查改，需要知识库（文本管理），向量数据库和搜索工具。
4. 针对于文档问答的增啥查改，需要文档知识库（文档管理），向量数据库（文档解析和切分然后转为向量）和搜索工具。
5. 针对于对话，需要大模型接口，以及问答的模板。
6. 搜索工具这里分为全文搜索和相似度计算搜索，合并之后进行重排序
7. 向量数据库这里需要文档解析功能，文本嵌入模型。

![2025-09-11_10-59.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/90f6fee4-b751-4439-87fd-bb8d2fe332e7.jpeg)

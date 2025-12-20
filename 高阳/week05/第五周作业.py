用RAG技术实现政企问答项目的流程

一、分析项目和调研
基于项目的需求，我将项目分为以下流程：
1、数据接入：知识来源清洗（PDF、Word、网页、数据库、API）
2、向量化存储：Embedding + ES
3、检索增强生成（RAG）：通过检索内容增强Prompt
4、问答引擎：基于 OpenAI API（或本地模型）

二、数据接入和清洗
用python的markdown库读取MD文件，用unstructured、pymupdf处理pdf文件，用python-docx处理word文件，网页信息用beautifulsoup。
    
三、向量化存储 
    可以使用huggingface模型如bge-small-zh-v1.5等来生成embedding，生成向量之后，构建索引并持久化到es中
 
四、构建RAG流程
当用户输入问题后，我们首先对问题生成Embedding，然后在用es检索最相关的文档,把检索结果和用户问题拼接为 Prompt，送给大模型

五、问答引擎
基于 OpenAI API或者本地部署的大模型，搭建api服务























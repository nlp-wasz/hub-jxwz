---
typora-root-url: ./assets
---

1、产品文档RAG介绍

![image-20250911214857334](/image-20250911214857334.png)

环境配置

配置elasticsearch

```
elasticsearch:
  host: "localhost"
  port: 9200
  scheme: "http"
  username: ""
  password: ""
```

- 存储文档内容和向量表示

- 提供全文检索和向量检索能力

- 支持混合检索策略

 配置sqlite数据库

```
database:
  engine: "sqlite"
  path: "rag.db"
  host: "localhost"
  port: 3306
  username: ""
  password: ""
```

- 管理知识库和文档的元数据

- 存储文档的位置、标题、类型等信息

- 通过ID与Elasticsearch建立关联

工作流程

### 知识库管理

- 用户创建知识库（如"政策法规库"、"产品手册库"）

- SQLite存储知识库的基本信息（ID、标题、分类等）

- 知识库ID作为后续文档分类的基础

### 2. 文档上传与处理

通过 POST /v1/document 接口处理：

1. 文档元数据处理

   - 验证知识库是否存在（SQLite查询）

   - 创建文档记录，获取自动生成的document_id

   - 保存文件到本地文件系统

   - 更新文件路径信息

1. PDF文档内容处理（后台任务）

   - 使用pdfplumber打开PDF文件

   - 逐页提取文本内容

   - 前4页内容作为文档摘要

   - 每页内容进行向量化

   - 存储页面级chunk到Elasticsearch

   - 页面内容分块处理

     - 按chunk_size(256)和chunk_overlap(20)分割

     - 对分块内容进行向量化

     - 存储分块chunk到Elasticsearch

- 存储文档元信息到document_meta索引

- ### 查询处理流程

  通过 POST /chat 接口处理：

  用户指定知识库

  - 用户提供knowledge_id，指定在哪个知识库中搜索

  - 这个ID来自SQLite，用于过滤Elasticsearch检索结果

  混合检索策略

  - 全文检索：使用BM25算法匹配关键词

  - 向量检索：计算语义相似度

  - 两者都使用knowledge_id进行过滤，确保只检索指定知识库的内容

  结果处理与回答生成

  - 融合检索结果(多项检测拿到相似值最高的结果)

  - 可选的重排序优化(通过 bge-reranker-base模型)

  - 构建RAG提示词

  - 调用LLM生成回答
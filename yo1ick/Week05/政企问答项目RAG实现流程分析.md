
## RAG 实现流程

### 1. 知识库与文档管理
- 使用 SQLite/MySQL 存储知识库和文档的元数据信息
- 通过 REST API 实现知识库和文档的增删改查操作
- 支持 PDF 等格式文档上传和解析

### 2. 文档处理与存储
- 文档上传后，系统在后台提取内容：
  - 对于 PDF 文件，使用 pdfplumber 提取文本内容
  - 将每页内容存储到 Elasticsearch 的 chunk_info 索引中
  - 同时将文本分块（chunk）处理，支持重叠分块
  - 为每个文本块生成嵌入向量并存储

### 3. 检索流程
当用户提问时，系统执行以下检索步骤：

1. **双路检索**：
   - 关键词检索（BM25）：在 Elasticsearch 中执行全文搜索
   - 语义检索：使用嵌入模型将查询编码为向量，进行向量相似度搜索

2. **结果融合**：
   - 使用 Reciprocal Rank Fusion (RRF) 算法融合两种检索方式的结果
   - 可配置是否使用重排序模型对结果进一步优化

3. **上下文构建**：
   - 从检索结果中提取最相关的文本块
   - 构建包含检索结果的提示模板

### 4. 生成回答
- 将检索到的相关文档内容与用户问题组合成提示
- 调用大语言模型（GLM-4-AIR）生成最终回答
- 支持调整温度、top_p 等生成参数

## 系统架构特点

1. **多模型支持**：
   - 嵌入模型：BGE-small-zh-v1.5
   - 重排序模型：BGE-reranker-base
   - LLM：GLM-4-AIR（通过 API 调用）

2. **灵活配置**：
   - 通过 config.yaml 文件配置各种参数
   - 支持启用/禁用嵌入检索、重排序、RRF 等功能

3. **扩展性**：
   - 模块化设计，易于添加新的文档解析器
   - 支持多种数据库后端（SQLite、MySQL）
   - 可扩展支持更多嵌入和重排序模型

## 安装与运行

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 Elasticsearch 并确保运行
3. 更新 config.yaml 中的相关配置
4. 运行主服务：`python main.py`
5. 通过 API 端点进行知识库管理和问答

## API 端点

- `GET/POST/DELETE /v1/knowledge_base` - 知识库管理
- `GET/POST/DELETE /v1/document` - 文档管理
- `POST /v1/embedding` - 文本嵌入服务
- `POST /v1/rerank` - 重排序服务
- `POST /chat` - 问答服务

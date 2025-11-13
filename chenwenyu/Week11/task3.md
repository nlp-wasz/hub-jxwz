基于Milvus设计多模态RAG的数据结构需要精心规划。以下是完整的数据结构设计方案：

## 1. 集合设计策略

### 分模态集合
```python
# 按模态分集合
collections = {
    "text": "multimodal_text_collection",
    "image": "multimodal_image_collection", 
    "table": "multimodal_table_collection",
    "audio": "multimodal_audio_collection",
    "video": "multimodal_video_collection"
}
```


## 2. 分模态集合的Schema设计

### 2.1 文本集合 multimodal_text_collection
```python
text_fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="page_number", dtype=DataType.INT32),
    FieldSchema(name="token_count", dtype=DataType.INT32),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
```

### 3.2 图像集合 multimodal_image_collection
```python
image_fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="visual_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="ocr_text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=32),  # "1920x1080"
    FieldSchema(name="format", dtype=DataType.VARCHAR, max_length=16),  # "png", "jpg"
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
```

### 3.3 表格集合 multimodal_table_collection
```python
table_fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="table_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="table_json", dtype=DataType.VARCHAR, max_length=65535),  # 表格结构化数据
    FieldSchema(name="table_summary", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="headers", dtype=DataType.JSON),  # 列头信息
    FieldSchema(name="row_count", dtype=DataType.INT32),
    FieldSchema(name="col_count", dtype=DataType.INT32),
    FieldSchema(name="data_types", dtype=DataType.JSON),  # 列数据类型
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
```

## 3. 索引策略

### 3.1 向量索引配置
```python
# 多模态嵌入向量索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",  # 或 "L2"
    "params": {"nlist": 2048}
}

# 文本嵌入向量索引  
text_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}

# 图像嵌入向量索引
image_index_params = {
    "index_type": "IVF_PQ", 
    "metric_type": "L2",
    "params": {"nlist": 1024, "m": 16, "nbits": 8}
}
```

### 3.2 标量字段索引
```python
# 为常用查询字段创建标量索引
scalar_index_params = {
    "index_type": "Trie"  # 用于VARCHAR字段的前缀搜索
}

# 需要索引的字段
index_fields = ["document_id", "modality", "partition_key", "created_time"]
```

## 4. 数据插入示例

```python
# 文本数据插入
text_data = {
    "chunk_id": "text_chunk_001",
    "document_id": "doc_123",
    "modality": "text",
    "multimodal_embedding": text_embedding_vector,
    "content": "这是文本内容...",
    "metadata": {
        "section": "引言",
        "page": 1,
        "language": "zh-CN",
        "importance": 0.8
    },
    "partition_key": "doc_123"
}

# 图像数据插入
image_data = {
    "chunk_id": "image_chunk_001", 
    "document_id": "doc_123",
    "modality": "image",
    "multimodal_embedding": image_embedding_vector,
    "content": "图片的文本描述...",
    "image_data": "base64_encoded_image_or_path",
    "metadata": {
        "caption": "系统生成的图片描述",
        "ocr_text": "识别出的文字内容",
        "image_type": "chart",
        "confidence": 0.95
    },
    "partition_key": "doc_123"
}

# 表格数据插入
table_data = {
    "chunk_id": "table_chunk_001",
    "document_id": "doc_123", 
    "modality": "table",
    "multimodal_embedding": table_embedding_vector,
    "content": "表格的文本摘要...",
    "table_data": json.dumps({
        "headers": ["姓名", "年龄", "城市"],
        "rows": [
            ["张三", "25", "北京"],
            ["李四", "30", "上海"]
        ]
    }),
    "metadata": {
        "table_type": "data_table",
        "has_header": True,
        "data_quality": 0.9
    },
    "partition_key": "doc_123"
}
```

## 6. 查询示例

### 6.1 多模态混合检索
```python
# 统一向量搜索
search_params = {
    "data": [query_embedding],  # 查询向量
    "anns_field": "multimodal_embedding",
    "param": {"metric_type": "COSINE", "params": {"nprobe": 32}},
    "limit": 10,
    "expr": "modality in ['text', 'image']"  # 模态过滤
}

# 带元数据过滤的检索
filter_expr = """
    modality == 'text' and 
    metadata['language'] == 'zh-CN' and  
    created_time > 1700000000
"""
```

### 6.2 跨模态检索
```python
# 用文本搜索图片
image_search_params = {
    "data": [text_embedding_for_image_search],
    "anns_field": "multimodal_embedding", 
    "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
    "limit": 5,
    "expr": "modality == 'image'"
}
```

## 7. 分区策略

```python
# 按文档分区（推荐）
partition_names = [f"doc_{doc_id}" for doc_id in document_ids]

# 按时间分区  
partition_names = [f"month_2024_{str(i).zfill(2)}" for i in range(1, 13)]

# 按模态分区
partition_names = ["text", "image", "table", "audio", "video"]
```

## 8. 性能优化建议

1. **向量维度统一**：使用统一的多模态嵌入模型确保向量空间一致
2. **分区管理**：根据查询模式设计分区策略
3. **索引调优**：根据数据量和查询延迟要求调整索引参数
4. **批量操作**：使用批量插入和查询提高吞吐量
5. **内存管理**：合理配置Milvus节点资源，使用SSD提升性能

这样的设计可以高效支持多模态RAG的存储和检索需求，同时保持良好的扩展性和性能。
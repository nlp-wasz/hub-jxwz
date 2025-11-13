# 【作业3】Milvus 结构

```python
# 【作业1 见】因为 sqlite中 knowledge表 和 document表 结构已经包含 文档所需信息
# 因此 Milvus中只需存储 图文编码结果 以及 knowledge表 和 document表 的ID字段，即可根据ID字段查询检索结果的具体信息

# Collection： multimodel_rag
```

| 字段名称           | 字段类型           | 默认值          | 描述 |
| ------------------ | ------------------ | --------------- | ---- |
| primary_key        | INT64              | The Primary Key |      |
| text_clip_feature  | FLOAT_VECTOR (512) |                 |      |
| image_clip_feature | FLOAT_VECTOR (512) |                 |      |
| text_bge_feature   | FLOAT_VECTOR (512) |                 |      |
| knowledge_id       | INT16              |                 |      |
| document_id        |                    | INT16           |      |


# task3

## Milvus多模态数据结构设计

### 核心集合设计原则

Milvus存储多模态数据需遵循"**模态分离+关联索引**"原则，将文本、图像、表格数据拆分为基础集合与关联集合，通过 `entity_id` 实现跨模态关联。所有向量字段统一使用**FloatVector**类型，维度对齐至1024维（基于CLIP模型输出），元数据字段采用结构化设计确保检索效率。

### 1. 文本数据集合（text_entities）

| 字段名 | 数据类型 | 主键 | 索引 | 描述 |
| --------- | ------------ | ------ | ------ | ------ |
| text_id | Int64 | 是 | 主键索引 | 文本实体唯一标识，自增ID |
| entity_id | String | 否 | 普通索引 | 跨模态关联ID（与图像/表格共享） |
| content | String | 否 | - | 原始文本内容（最大长度65535字符） |
| vector | FloatVector(1024) | 否 | IVF_FLAT | 文本向量，使用Sentence-BERT生成 |
| doc_id | String | 否 | 普通索引 | 关联文档ID |
| chunk_num | Int32 | 否 | - | 文档内分块序号 |
| metadata | JSON | 否 | - | 扩展元数据（如关键词、时间戳） |
| created_at | DateTime | 否 | 普通索引 | 创建时间（UTC） |

**索引配置**：
```python
{
  "index_type": "IVF_FLAT",
  "metric_type": "COSINE",
  "params": {"nlist": 1024}
}
```

### 2. 图像数据集合（image_entities）

| 字段名 | 数据类型 | 主键 | 索引 | 描述 |
| --------- | ------------ | ------ | ------ | ------ |
| image_id | Int64 | 是 | 主键索引 | 图像实体唯一标识，自增ID |
| entity_id | String | 否 | 普通索引 | 跨模态关联ID（与文本/表格共享） |
| image_url | String | 否 | - | 图像存储路径（支持S3/OSS URL） |
| vector | FloatVector(1024) | 否 | IVF_FLAT | 图像向量，使用CLIP ViT-L/14生成 |
| doc_id | String | 否 | 普通索引 | 关联文档ID |
| page_num | Int32 | 否 | - | 文档内页码 |
| bbox | JSON | 否 | - | 图像在文档中的坐标（x,y,w,h） |
| metadata | JSON | 否 | - | 图像描述（如OCR文本、物体检测标签） |
| created_at | DateTime | 否 | 普通索引 | 创建时间（UTC） |

**示例metadata**：
```json
{
  "ocr_text": "产品尺寸：20x30x40cm",
  "objects": ["product", "label"],
  "resolution": "1920x1080"
}
```

### 3. 表格数据集合（table_entities）

| 字段名 | 数据类型 | 主键 | 索引 | 描述 |
| --------- | ------------ | ------ | ------ | ------ |
| table_id | Int64 | 是 | 主键索引 | 表格实体唯一标识，自增ID |
| entity_id | String | 否 | 普通索引 | 跨模态关联ID（与文本/图像共享） |
| table_data | JSON | 否 | - | 结构化表格数据（行列数组） |
| vector | FloatVector(1024) | 否 | IVF_FLAT | 表格向量，使用TableBERT生成 |
| doc_id | String | 否 | 普通索引 | 关联文档ID |
| page_num | Int32 | 否 | - | 文档内页码 |
| table_title | String | 否 | - | 表格标题（若存在） |
| row_count | Int32 | 否 | - | 表格行数 |
| col_count | Int32 | 否 | - | 表格列数 |
| created_at | DateTime | 否 | 普通索引 | 创建时间（UTC） |

**示例table_data**：
```json
{
  "headers": ["参数", "数值", "单位"],
  "rows": [
    ["长度", "20", "cm"],
    ["宽度", "30", "cm"],
    ["高度", "40", "cm"]
  ]
}
```

### 4. 跨模态关联集合（cross_modal_links）

| 字段名 | 数据类型 | 主键 | 索引 | 描述 |
| --------- | ------------ | ------ | ------ | ------ |
| link_id | Int64 | 是 | 主键索引 | 关联记录唯一标识 |
| entity_id | String | 否 | 普通索引 | 跨模态关联ID |
| modalities | Array[String] | 否 | - | 包含的模态类型（如["text","image"]） |
| doc_id | String | 否 | 普通索引 | 关联文档ID |
| confidence | Float32 | 否 | - | 模态关联置信度（0-1.0） |
| created_at | DateTime | 否 | - | 创建时间（UTC） |

### 索引优化策略

- **向量索引**：所有向量字段采用**IVF_FLAT**索引（召回率优先），nlist参数设为1024（适用于百万级数据量）
- **复合索引**：对 `(doc_id, entity_id)` 创建复合索引，加速跨模态联合查询
- **分区键**：按 `doc_id` 进行分区，支持按文档粒度的数据隔离与删除

### 数据写入流程

- 文档上传后解析为文本块、图像、表格三种实体
- 分别生成向量并写入对应集合
- 生成全局唯一 `entity_id` 建立跨模态关联
- 在cross_modal_links集合记录关联关系

### 查询示例（Python SDK）
```python
# 跨模态检索示例：根据文本向量查询相关图像
from pymilvus import Collection

text_col = Collection("text_entities")
image_col = Collection("image_entities")

# 1. 查询文本向量对应的entity_id
text_res = text_col.search(
  data=[query_vector],
  anns_field="vector",
  param={"nprobe": 16},
  limit=5,
  output_fields=["entity_id"]
)

entity_ids = [hit.entity.get("entity_id") for hit in text_res[0]]

# 2. 通过entity_id查询关联图像
image_res = image_col.query(
  expr=f"entity_id in {entity_ids}",
  output_fields=["image_url", "metadata"]
)
```

### 存储容量估算

- 单条文本实体：约2KB（含向量4KB+文本内容）
- 单条图像实体：约5KB（含向量4KB+URL及元数据）
- 单条表格实体：约6KB（含向量4KB+结构化数据）
- **百万级数据量**：总存储约15GB（含索引），建议使用Milvus 2.3+版本配合对象存储（MinIO/S3）存储原始文件。

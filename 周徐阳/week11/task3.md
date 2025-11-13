# Milvus多模态RAG数据结构设计指南

## 目录
- [1. Milvus基础概念](#1-milvus基础概念)
- [2. 多模态存储架构设计](#2-多模态存储架构设计)
- [3. Collection设计方案](#3-collection设计方案)
- [4. 字段Schema详细设计](#4-字段schema详细设计)
- [5. 索引策略](#5-索引策略)
- [6. 完整代码实现](#6-完整代码实现)
- [7. 查询与检索策略](#7-查询与检索策略)
- [8. 最佳实践与优化](#8-最佳实践与优化)

---

## 1. Milvus基础概念

### 1.1 核心概念

| 概念 | 说明 | 类似概念 |
|-----|------|---------|
| **Collection** | 数据集合，类似表 | 关系数据库的Table |
| **Field** | 字段，包含向量字段和标量字段 | 列 |
| **Entity** | 一条记录 | 行 |
| **Vector Field** | 存储向量的字段 | 向量列 |
| **Scalar Field** | 存储标量数据（字符串、数字等） | 普通列 |
| **Partition** | Collection的分区 | 表分区 |
| **Index** | 向量索引（IVF、HNSW等） | 数据库索引 |

### 1.2 Milvus支持的数据类型

```python
# 标量字段类型
DataType.INT8       # 8位整数
DataType.INT16      # 16位整数
DataType.INT32      # 32位整数
DataType.INT64      # 64位整数
DataType.FLOAT      # 单精度浮点
DataType.DOUBLE     # 双精度浮点
DataType.VARCHAR    # 变长字符串
DataType.JSON       # JSON类型（Milvus 2.2+）
DataType.ARRAY      # 数组类型（Milvus 2.4+）

# 向量字段类型
DataType.FLOAT_VECTOR   # 浮点向量
DataType.BINARY_VECTOR  # 二值向量
```

---

## 2. 多模态存储架构设计

### 2.1 架构选择对比

#### 方案A: 单Collection统一存储（推荐）

```
┌─────────────────────────────────────────────────────────┐
│             Multimodal_Collection                        │
├─────────────────────────────────────────────────────────┤
│  content_type: "text" | "image" | "table"               │
│  text_embedding: [768维向量]                             │
│  image_embedding: [512维向量]                            │
│  content: 原始内容或引用                                  │
│  metadata: {详细元数据}                                   │
└─────────────────────────────────────────────────────────┘
```

**优点**：
- 统一管理，简化架构
- 便于跨模态检索
- 减少维护成本

**缺点**：
- 某些字段对特定类型数据无效（占用空间）
- 查询可能需要过滤content_type

#### 方案B: 多Collection分离存储

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Text_Collection  │  │ Image_Collection │  │ Table_Collection │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ text_embedding   │  │ image_embedding  │  │ table_embedding  │
│ content          │  │ image_url        │  │ table_data       │
│ metadata         │  │ caption          │  │ metadata         │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**优点**：
- 字段专用，节省空间
- 针对性优化索引
- 便于独立扩展

**缺点**：
- 跨模态检索复杂
- 管理成本高
- 需要结果合并逻辑

#### 方案C: 混合方案（推荐用于大规模）

```
┌─────────────────────────────────────────────────────────┐
│             Unified_Index_Collection                     │
│  (存储所有模态的统一向量表示)                              │
├─────────────────────────────────────────────────────────┤
│  unified_embedding: [768维向量]                          │
│  content_type: "text" | "image" | "table"               │
│  reference_id: 指向具体Collection的ID                     │
└─────────────────────────────────────────────────────────┘
              ↓ reference_id 指向
┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│Text_Details  │  │Image_Details  │  │Table_Details  │
│(无向量索引)   │  │(无向量索引)    │  │(无向量索引)    │
└──────────────┘  └───────────────┘  └───────────────┘
```

**优点**：
- 检索高效（只查一个Collection）
- 存储优化（详细信息分离）
- 便于扩展

**缺点**：
- 需要二次查询获取详情
- 架构相对复杂

### 2.2 推荐方案选择

| 数据规模 | 推荐方案 | 理由 |
|---------|---------|------|
| < 100万条 | 方案A（单Collection） | 简单高效，易于维护 |
| 100万-1000万 | 方案C（混合方案） | 平衡性能和灵活性 |
| > 1000万条 | 方案B+分片 | 独立优化，水平扩展 |

---

## 3. Collection设计方案

### 3.1 方案A: 单Collection统一存储（详细设计）

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# Collection Schema定义
def create_multimodal_collection_schema():
    """
    创建统一的多模态Collection Schema
    """
    fields = [
        # ===== 主键字段 =====
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=False,
            description="唯一标识符，格式：{type}_{doc_id}_{chunk_id}"
        ),

        # ===== 内容类型字段 =====
        FieldSchema(
            name="content_type",
            dtype=DataType.VARCHAR,
            max_length=20,
            description="内容类型：text|image|table"
        ),

        # ===== 文本向量字段 =====
        FieldSchema(
            name="text_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,  # BGE模型维度
            description="文本向量（对于image/table也会生成描述文本的向量）"
        ),

        # ===== 图像向量字段 =====
        FieldSchema(
            name="image_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=512,  # CLIP ViT-B/32维度
            description="图像向量（仅image类型有效，其他类型填充零向量）"
        ),

        # ===== 内容字段 =====
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="文本内容 或 图片描述 或 表格markdown"
        ),

        FieldSchema(
            name="content_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="图片URL或其他资源URL"
        ),

        # ===== 文档关联字段 =====
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="所属文档ID"
        ),

        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT32,
            description="在文档中的位置索引"
        ),

        # ===== 元数据字段（JSON） =====
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="灵活的元数据存储"
        ),

        # ===== 时间戳字段 =====
        FieldSchema(
            name="created_at",
            dtype=DataType.INT64,
            description="创建时间戳（毫秒）"
        ),

        # ===== 文本特定字段 =====
        FieldSchema(
            name="token_count",
            dtype=DataType.INT32,
            description="文本token数量"
        ),

        # ===== 图像特定字段 =====
        FieldSchema(
            name="image_width",
            dtype=DataType.INT32,
            description="图片宽度"
        ),

        FieldSchema(
            name="image_height",
            dtype=DataType.INT32,
            description="图片高度"
        ),

        # ===== 表格特定字段 =====
        FieldSchema(
            name="table_rows",
            dtype=DataType.INT32,
            description="表格行数"
        ),

        FieldSchema(
            name="table_cols",
            dtype=DataType.INT32,
            description="表格列数"
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="多模态RAG统一存储Collection",
        enable_dynamic_field=True  # 允许动态字段
    )

    return schema
```

### 3.2 metadata字段详细设计

```python
# metadata JSON结构设计
metadata_schema = {
    # ===== 通用字段 =====
    "source": {
        "type": "string",
        "description": "数据来源：upload|crawl|api",
        "example": "upload"
    },
    "title": {
        "type": "string",
        "description": "文档标题",
        "example": "产品技术手册"
    },
    "author": {
        "type": "string",
        "description": "作者",
        "example": "张三"
    },
    "category": {
        "type": "string",
        "description": "分类",
        "example": "技术文档"
    },
    "tags": {
        "type": "array",
        "items": "string",
        "description": "标签列表",
        "example": ["AI", "RAG", "多模态"]
    },

    # ===== 文本特定元数据 =====
    "text_metadata": {
        "language": "zh",
        "section": "第3章",
        "page_number": 15,
        "heading_level": 2,
        "summary": "本段介绍了..."
    },

    # ===== 图像特定元数据 =====
    "image_metadata": {
        "format": "jpg|png|webp",
        "caption": "图片说明文字",
        "alt_text": "替代文本",
        "ocr_text": "OCR提取的文本",
        "detected_objects": ["person", "car"],
        "scene_type": "indoor|outdoor",
        "position_in_doc": "before_paragraph_5",
        "related_text": "图片前后的文本内容"
    },

    # ===== 表格特定元数据 =====
    "table_metadata": {
        "table_type": "data|comparison|financial",
        "headers": ["列1", "列2", "列3"],
        "caption": "表1：销售数据统计",
        "has_merged_cells": False,
        "csv_format": "column1,column2\\nval1,val2",
        "markdown_format": "| col1 | col2 |\\n|--|--|",
        "summary": "表格内容摘要"
    },

    # ===== 关系元数据 =====
    "relationships": {
        "parent_chunk_id": "chunk_parent_001",
        "child_chunk_ids": ["chunk_child_001", "chunk_child_002"],
        "related_image_ids": ["img_001", "img_002"],
        "related_table_ids": ["table_001"]
    }
}
```

### 3.3 实体数据示例

#### 示例1: 文本实体

```python
text_entity = {
    "id": "text_doc123_chunk001",
    "content_type": "text",
    "text_embedding": [0.123, -0.456, ...],  # 768维
    "image_embedding": [0.0] * 512,  # 零向量（占位）
    "content": "多模态RAG系统是一种结合了文本、图像等多种模态数据的检索增强生成系统...",
    "content_url": "",
    "document_id": "doc123",
    "chunk_index": 1,
    "metadata": {
        "source": "upload",
        "title": "多模态RAG技术白皮书",
        "category": "技术文档",
        "tags": ["RAG", "多模态", "AI"],
        "text_metadata": {
            "language": "zh",
            "section": "第2章",
            "page_number": 5,
            "summary": "介绍多模态RAG的基本概念"
        }
    },
    "created_at": 1704067200000,
    "token_count": 156,
    "image_width": 0,
    "image_height": 0,
    "table_rows": 0,
    "table_cols": 0
}
```

#### 示例2: 图像实体

```python
image_entity = {
    "id": "image_doc123_img001",
    "content_type": "image",
    "text_embedding": [0.234, -0.567, ...],  # 基于caption的向量
    "image_embedding": [0.345, 0.678, ...],  # 512维CLIP向量
    "content": "这是一张展示RAG系统架构的流程图，包含数据摄入、向量化、检索、生成四个主要模块",
    "content_url": "https://storage.example.com/images/doc123_img001.jpg",
    "document_id": "doc123",
    "chunk_index": 2,
    "metadata": {
        "source": "upload",
        "title": "多模态RAG技术白皮书",
        "category": "技术文档",
        "tags": ["架构图", "流程图"],
        "image_metadata": {
            "format": "jpg",
            "caption": "图1：RAG系统架构图",
            "ocr_text": "数据摄入 -> 向量化 -> 存储 -> 检索 -> 生成",
            "detected_objects": ["diagram", "flowchart"],
            "scene_type": "diagram",
            "position_in_doc": "after_paragraph_3",
            "related_text": "如图1所示，RAG系统主要包含..."
        },
        "relationships": {
            "related_text_chunks": ["text_doc123_chunk001", "text_doc123_chunk002"]
        }
    },
    "created_at": 1704067200000,
    "token_count": 0,
    "image_width": 1920,
    "image_height": 1080,
    "table_rows": 0,
    "table_cols": 0
}
```

#### 示例3: 表格实体

```python
table_entity = {
    "id": "table_doc123_tbl001",
    "content_type": "table",
    "text_embedding": [0.456, -0.789, ...],  # 基于表格内容的向量
    "image_embedding": [0.0] * 512,  # 零向量（或表格截图的向量）
    "content": """
| 模型 | 参数量 | 准确率 | 推理速度 |
|------|--------|--------|----------|
| GPT-4 | 1.7T | 95.2% | 50 tokens/s |
| Claude-3 | Unknown | 94.8% | 80 tokens/s |
| Qwen-Max | 72B | 92.5% | 120 tokens/s |
""",
    "content_url": "",
    "document_id": "doc123",
    "chunk_index": 3,
    "metadata": {
        "source": "upload",
        "title": "多模态RAG技术白皮书",
        "category": "技术文档",
        "tags": ["性能对比", "模型参数"],
        "table_metadata": {
            "table_type": "comparison",
            "headers": ["模型", "参数量", "准确率", "推理速度"],
            "caption": "表1：主流大模型性能对比",
            "has_merged_cells": False,
            "csv_format": "模型,参数量,准确率,推理速度\\nGPT-4,1.7T,95.2%,50 tokens/s\\n...",
            "summary": "对比了GPT-4、Claude-3、Qwen-Max三个模型的性能指标"
        },
        "relationships": {
            "related_text_chunks": ["text_doc123_chunk004"]
        }
    },
    "created_at": 1704067200000,
    "token_count": 0,
    "image_width": 0,
    "image_height": 0,
    "table_rows": 3,
    "table_cols": 4
}
```

---

## 4. 字段Schema详细设计

### 4.1 字段选择策略

#### 必需字段（所有实体）
```python
required_fields = [
    "id",              # 主键
    "content_type",    # 类型标识
    "text_embedding",  # 文本向量（必有）
    "content",         # 内容
    "document_id",     # 文档关联
    "metadata",        # 元数据
    "created_at"       # 时间戳
]
```

#### 可选字段（按需使用）
```python
optional_fields = {
    "image_embedding": "仅当需要图像检索时",
    "content_url": "仅当有外部资源时",
    "token_count": "仅text类型",
    "image_width/height": "仅image类型",
    "table_rows/cols": "仅table类型"
}
```

### 4.2 向量维度选择

| 模态 | 推荐模型 | 向量维度 | 说明 |
|-----|---------|---------|------|
| **文本** | BGE-large-zh | 1024 | 中文效果好 |
|  | M3E-large | 1024 | 开源中文 |
|  | text-embedding-3-large | 3072 | OpenAI最新 |
| **图像** | CLIP ViT-L/14 | 768 | 平衡性能 |
|  | CLIP ViT-B/32 | 512 | 速度快 |
|  | SigLIP | 1152 | 最新SOTA |
| **多模态** | BLIP-2 | 768 | 统一编码 |
|  | Qwen-VL | 4096 | 强大但慢 |

**重要提示**：
- 如果使用统一多模态模型（如BLIP-2），可以只用一个向量字段
- 如果分别编码，需要两个向量字段
- 向量维度越高，存储和检索成本越高

### 4.3 字段数据类型最佳实践

```python
# VARCHAR长度建议
field_length_recommendations = {
    "id": 64,              # 足够存储UUID或自定义ID
    "content_type": 20,    # "text"/"image"/"table"
    "content": 65535,      # 最大64KB（Milvus限制）
    "content_url": 512,    # 标准URL长度
    "document_id": 64,     # 与id保持一致
}

# INT类型选择
int_type_selection = {
    "chunk_index": "INT32",      # 足够表示百万级chunks
    "token_count": "INT32",      # 最大2^31-1
    "image_width/height": "INT32",  # 足够表示超高清
    "table_rows/cols": "INT16",  # 通常不超过32K
    "created_at": "INT64"        # 时间戳毫秒需要64位
}
```

---

## 5. 索引策略

### 5.1 向量索引选择

```python
# 文本向量索引（高维）
text_index_params = {
    "metric_type": "IP",  # Inner Product（内积）或 "L2"（欧氏距离）
    "index_type": "HNSW",  # 高性能索引
    "params": {
        "M": 16,           # 每层最大连接数（8-64）
        "efConstruction": 200  # 构建时搜索深度（100-500）
    }
}

# 图像向量索引（中维）
image_index_params = {
    "metric_type": "COSINE",  # 余弦相似度
    "index_type": "IVF_FLAT",  # 适合中等规模
    "params": {
        "nlist": 1024  # 聚类中心数量
    }
}

# 大规模数据索引
large_scale_index_params = {
    "metric_type": "IP",
    "index_type": "IVF_PQ",  # 乘积量化
    "params": {
        "nlist": 2048,
        "m": 8,  # PQ分段数
        "nbits": 8  # 每段比特数
    }
}
```

### 5.2 标量字段索引

```python
# 为常用过滤字段创建标量索引
scalar_index_fields = [
    "content_type",   # 按类型过滤
    "document_id",    # 按文档过滤
    "created_at"      # 按时间范围过滤
]

# Milvus会自动为这些字段创建索引
# 通过构建Collection时指定或后续添加
```

### 5.3 索引性能对比

| 索引类型 | 检索速度 | 内存占用 | 准确率 | 适用场景 |
|---------|---------|---------|--------|---------|
| **FLAT** | 慢 | 高 | 100% | 小规模（<10万） |
| **IVF_FLAT** | 中 | 中 | 95%+ | 中等规模（10万-100万） |
| **IVF_PQ** | 快 | 低 | 85%+ | 大规模（>100万） |
| **HNSW** | 很快 | 高 | 98%+ | 实时查询，内存充足 |

---

## 6. 完整代码实现

### 6.1 创建Collection

```python
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)

class MultimodalMilvusManager:
    """
    多模态Milvus管理器
    """
    def __init__(self, host="localhost", port="19530"):
        """初始化连接"""
        connections.connect(
            alias="default",
            host=host,
            port=port
        )

    def create_multimodal_collection(
        self,
        collection_name="multimodal_rag",
        text_dim=768,
        image_dim=512
    ):
        """
        创建多模态Collection
        """
        # 检查Collection是否已存在
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists")
            return Collection(collection_name)

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR,
                       max_length=64, is_primary=True),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR,
                       max_length=20),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR,
                       dim=text_dim),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR,
                       dim=image_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR,
                       max_length=65535),
            FieldSchema(name="content_url", dtype=DataType.VARCHAR,
                       max_length=512),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR,
                       max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="token_count", dtype=DataType.INT32),
            FieldSchema(name="image_width", dtype=DataType.INT32),
            FieldSchema(name="image_height", dtype=DataType.INT32),
            FieldSchema(name="table_rows", dtype=DataType.INT32),
            FieldSchema(name="table_cols", dtype=DataType.INT32),
        ]

        # 创建Schema
        schema = CollectionSchema(
            fields=fields,
            description="Multimodal RAG Collection",
            enable_dynamic_field=True
        )

        # 创建Collection
        collection = Collection(
            name=collection_name,
            schema=schema
        )

        print(f"Collection {collection_name} created successfully")
        return collection

    def create_indexes(self, collection_name="multimodal_rag"):
        """
        创建索引
        """
        collection = Collection(collection_name)

        # 文本向量索引
        text_index = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(
            field_name="text_embedding",
            index_params=text_index
        )
        print("Text embedding index created")

        # 图像向量索引
        image_index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="image_embedding",
            index_params=image_index
        )
        print("Image embedding index created")

        # 加载Collection到内存
        collection.load()
        print("Collection loaded into memory")

        return collection
```

### 6.2 插入数据

```python
import time
import uuid

class MultimodalDataInserter:
    """
    多模态数据插入器
    """
    def __init__(self, collection_name="multimodal_rag"):
        self.collection = Collection(collection_name)

    def insert_text_chunk(
        self,
        content: str,
        text_embedding: list,
        document_id: str,
        chunk_index: int,
        metadata: dict = None
    ):
        """
        插入文本块
        """
        entity = {
            "id": f"text_{document_id}_{chunk_index}",
            "content_type": "text",
            "text_embedding": text_embedding,
            "image_embedding": [0.0] * 512,  # 零向量占位
            "content": content,
            "content_url": "",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "created_at": int(time.time() * 1000),
            "token_count": len(content.split()),
            "image_width": 0,
            "image_height": 0,
            "table_rows": 0,
            "table_cols": 0
        }

        self.collection.insert([entity])
        return entity["id"]

    def insert_image(
        self,
        caption: str,
        image_url: str,
        text_embedding: list,
        image_embedding: list,
        document_id: str,
        chunk_index: int,
        width: int,
        height: int,
        metadata: dict = None
    ):
        """
        插入图像
        """
        entity = {
            "id": f"image_{document_id}_{chunk_index}",
            "content_type": "image",
            "text_embedding": text_embedding,  # 基于caption
            "image_embedding": image_embedding,
            "content": caption,
            "content_url": image_url,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "created_at": int(time.time() * 1000),
            "token_count": 0,
            "image_width": width,
            "image_height": height,
            "table_rows": 0,
            "table_cols": 0
        }

        self.collection.insert([entity])
        return entity["id"]

    def insert_table(
        self,
        table_content: str,
        text_embedding: list,
        document_id: str,
        chunk_index: int,
        rows: int,
        cols: int,
        metadata: dict = None
    ):
        """
        插入表格
        """
        entity = {
            "id": f"table_{document_id}_{chunk_index}",
            "content_type": "table",
            "text_embedding": text_embedding,
            "image_embedding": [0.0] * 512,
            "content": table_content,
            "content_url": "",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "created_at": int(time.time() * 1000),
            "token_count": 0,
            "image_width": 0,
            "image_height": 0,
            "table_rows": rows,
            "table_cols": cols
        }

        self.collection.insert([entity])
        return entity["id"]

    def batch_insert(self, entities: list):
        """
        批量插入
        """
        # 按字段组织数据
        data = {
            "id": [],
            "content_type": [],
            "text_embedding": [],
            "image_embedding": [],
            "content": [],
            "content_url": [],
            "document_id": [],
            "chunk_index": [],
            "metadata": [],
            "created_at": [],
            "token_count": [],
            "image_width": [],
            "image_height": [],
            "table_rows": [],
            "table_cols": []
        }

        for entity in entities:
            for key in data.keys():
                data[key].append(entity[key])

        # 批量插入
        self.collection.insert(list(data.values()))
        self.collection.flush()
        print(f"Inserted {len(entities)} entities")
```

### 6.3 查询与检索

```python
class MultimodalSearcher:
    """
    多模态检索器
    """
    def __init__(self, collection_name="multimodal_rag"):
        self.collection = Collection(collection_name)
        self.collection.load()

    def search_by_text(
        self,
        query_embedding: list,
        top_k: int = 10,
        content_types: list = None,
        filters: dict = None
    ):
        """
        文本向量检索
        """
        # 构建过滤表达式
        expr = self._build_filter_expr(content_types, filters)

        # 检索参数
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 100}  # HNSW搜索参数
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="text_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content_type", "content",
                          "content_url", "metadata", "document_id"]
        )

        return self._format_results(results)

    def search_by_image(
        self,
        query_embedding: list,
        top_k: int = 10,
        filters: dict = None
    ):
        """
        图像向量检索
        """
        expr = self._build_filter_expr(["image"], filters)

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 32}  # IVF搜索参数
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="image_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content_type", "content",
                          "content_url", "metadata", "image_width",
                          "image_height"]
        )

        return self._format_results(results)

    def hybrid_search(
        self,
        text_embedding: list,
        image_embedding: list = None,
        top_k: int = 10,
        text_weight: float = 0.6,
        image_weight: float = 0.4
    ):
        """
        混合检索（文本+图像）
        """
        # 文本检索
        text_results = self.search_by_text(
            query_embedding=text_embedding,
            top_k=top_k * 2
        )

        if image_embedding is None:
            return text_results[:top_k]

        # 图像检索
        image_results = self.search_by_image(
            query_embedding=image_embedding,
            top_k=top_k * 2
        )

        # 结果融合（加权）
        fused_results = self._fuse_results(
            text_results, image_results,
            text_weight, image_weight
        )

        return fused_results[:top_k]

    def _build_filter_expr(
        self,
        content_types: list = None,
        filters: dict = None
    ):
        """
        构建过滤表达式
        """
        expressions = []

        # 内容类型过滤
        if content_types:
            type_expr = " or ".join(
                [f'content_type == "{t}"' for t in content_types]
            )
            expressions.append(f"({type_expr})")

        # 其他过滤条件
        if filters:
            if "document_id" in filters:
                expressions.append(
                    f'document_id == "{filters["document_id"]}"'
                )

            if "date_range" in filters:
                start, end = filters["date_range"]
                expressions.append(
                    f"created_at >= {start} and created_at <= {end}"
                )

        return " and ".join(expressions) if expressions else None

    def _format_results(self, results):
        """
        格式化检索结果
        """
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.entity.get("id"),
                    "score": hit.score,
                    "content_type": hit.entity.get("content_type"),
                    "content": hit.entity.get("content"),
                    "content_url": hit.entity.get("content_url"),
                    "metadata": hit.entity.get("metadata"),
                    "document_id": hit.entity.get("document_id")
                })
        return formatted

    def _fuse_results(
        self,
        text_results: list,
        image_results: list,
        text_weight: float,
        image_weight: float
    ):
        """
        融合多路检索结果
        """
        # 使用RRF (Reciprocal Rank Fusion)
        k = 60
        scores = {}

        # 文本结果
        for rank, item in enumerate(text_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, {
                "score": 0,
                "item": item
            })
            scores[item_id]["score"] += text_weight / (k + rank + 1)

        # 图像结果
        for rank, item in enumerate(image_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, {
                "score": 0,
                "item": item
            })
            scores[item_id]["score"] += image_weight / (k + rank + 1)

        # 排序
        sorted_items = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["item"] for item in sorted_items]
```

### 6.4 数据管理

```python
class MultimodalDataManager:
    """
    数据管理器
    """
    def __init__(self, collection_name="multimodal_rag"):
        self.collection = Collection(collection_name)

    def delete_by_document_id(self, document_id: str):
        """
        删除指定文档的所有数据
        """
        expr = f'document_id == "{document_id}"'
        self.collection.delete(expr)
        print(f"Deleted entities with document_id: {document_id}")

    def delete_by_ids(self, ids: list):
        """
        按ID删除
        """
        id_list = '", "'.join(ids)
        expr = f'id in ["{id_list}"]'
        self.collection.delete(expr)
        print(f"Deleted {len(ids)} entities")

    def get_by_id(self, entity_id: str):
        """
        根据ID查询实体
        """
        expr = f'id == "{entity_id}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["*"]
        )
        return results[0] if results else None

    def get_collection_stats(self):
        """
        获取Collection统计信息
        """
        stats = self.collection.get_collection_stats()

        # 按类型统计
        type_counts = {}
        for content_type in ["text", "image", "table"]:
            expr = f'content_type == "{content_type}"'
            count = self.collection.query(
                expr=expr,
                output_fields=["count(*)"]
            )
            type_counts[content_type] = count

        return {
            "total_entities": stats["row_count"],
            "type_distribution": type_counts,
            "collection_name": self.collection.name
        }

    def update_metadata(self, entity_id: str, new_metadata: dict):
        """
        更新元数据（先删除后插入）
        """
        # 查询现有实体
        entity = self.get_by_id(entity_id)
        if not entity:
            print(f"Entity {entity_id} not found")
            return

        # 删除旧实体
        self.delete_by_ids([entity_id])

        # 更新metadata
        entity["metadata"].update(new_metadata)

        # 插入更新后的实体
        self.collection.insert([entity])
        print(f"Updated metadata for {entity_id}")
```

---

## 7. 查询与检索策略

### 7.1 检索模式设计

```python
class RetrievalStrategy:
    """
    检索策略管理
    """
    @staticmethod
    def text_only_retrieval(searcher, query_embedding, top_k=10):
        """
        纯文本检索
        """
        return searcher.search_by_text(
            query_embedding=query_embedding,
            top_k=top_k,
            content_types=["text"]
        )

    @staticmethod
    def image_only_retrieval(searcher, query_embedding, top_k=10):
        """
        纯图像检索
        """
        return searcher.search_by_image(
            query_embedding=query_embedding,
            top_k=top_k
        )

    @staticmethod
    def table_focused_retrieval(searcher, query_embedding, top_k=5):
        """
        表格优先检索
        """
        # 先检索表格
        table_results = searcher.search_by_text(
            query_embedding=query_embedding,
            top_k=top_k,
            content_types=["table"]
        )

        # 再检索相关文本
        text_results = searcher.search_by_text(
            query_embedding=query_embedding,
            top_k=top_k,
            content_types=["text"]
        )

        # 合并结果（表格优先）
        return table_results + text_results

    @staticmethod
    def multimodal_comprehensive_retrieval(
        searcher,
        text_embedding,
        image_embedding=None,
        top_k=10
    ):
        """
        全模态检索
        """
        results = {
            "text": searcher.search_by_text(
                query_embedding=text_embedding,
                top_k=top_k,
                content_types=["text"]
            ),
            "images": searcher.search_by_text(
                query_embedding=text_embedding,
                top_k=top_k // 2,
                content_types=["image"]
            ),
            "tables": searcher.search_by_text(
                query_embedding=text_embedding,
                top_k=top_k // 2,
                content_types=["table"]
            )
        }

        if image_embedding:
            results["visual_search"] = searcher.search_by_image(
                query_embedding=image_embedding,
                top_k=top_k // 2
            )

        return results
```

### 7.2 过滤查询示例

```python
# 示例1: 按文档ID过滤
results = searcher.search_by_text(
    query_embedding=embedding,
    top_k=10,
    filters={"document_id": "doc123"}
)

# 示例2: 按时间范围过滤
import time
start_time = int(time.mktime(time.strptime("2024-01-01", "%Y-%m-%d")) * 1000)
end_time = int(time.time() * 1000)

results = searcher.search_by_text(
    query_embedding=embedding,
    top_k=10,
    filters={"date_range": (start_time, end_time)}
)

# 示例3: 按内容类型和文档过滤
results = searcher.search_by_text(
    query_embedding=embedding,
    top_k=10,
    content_types=["text", "table"],
    filters={"document_id": "doc123"}
)
```

---

## 8. 最佳实践与优化

### 8.1 性能优化

#### 优化1: 批量操作

```python
# ❌ 不推荐：逐条插入
for entity in entities:
    collection.insert([entity])

# ✅ 推荐：批量插入
batch_size = 1000
for i in range(0, len(entities), batch_size):
    batch = entities[i:i+batch_size]
    collection.insert(batch)
collection.flush()  # 确保数据持久化
```

#### 优化2: 选择合适的索引

```python
# 小规模数据（<10万）
index_small = {
    "index_type": "FLAT",  # 精确搜索
    "metric_type": "IP"
}

# 中等规模（10万-100万）
index_medium = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 1024}
}

# 大规模（>100万）
index_large = {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 16, "efConstruction": 200}
}
```

#### 优化3: 查询参数调优

```python
# HNSW查询参数
search_params_hnsw = {
    "metric_type": "IP",
    "params": {
        "ef": 100  # 增大ef提高准确率（降低速度）
    }
}

# IVF查询参数
search_params_ivf = {
    "metric_type": "IP",
    "params": {
        "nprobe": 32  # 增大nprobe提高准确率（降低速度）
    }
}
```

### 8.2 存储优化

#### 优化1: 稀疏向量处理

```python
# 对于不需要的向量字段，使用零向量占位
def create_zero_vector(dim: int):
    """创建零向量"""
    return [0.0] * dim

# 文本实体不需要图像向量
text_entity = {
    ...
    "image_embedding": create_zero_vector(512)
}

# 或者使用动态字段（更节省空间）
# 但会增加查询复杂度
```

#### 优化2: 内容字段优化

```python
# 大文本内容截断
def truncate_content(content: str, max_length: int = 10000):
    """截断过长内容"""
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

# 或者将大内容存储在外部（S3、OSS等）
entity = {
    "content": truncate_content(large_content),
    "content_url": "s3://bucket/full_content.txt"
}
```

### 8.3 数据一致性

```python
# 使用事务性操作（Milvus 2.4+）
from pymilvus import db

# 创建session
session = db.create_session()

try:
    # 批量操作
    session.insert(entities)
    session.flush()
    session.commit()
except Exception as e:
    session.rollback()
    print(f"Error: {e}")
finally:
    session.close()
```

### 8.4 监控与维护

```python
class MilvusMonitor:
    """
    Milvus监控工具
    """
    @staticmethod
    def check_collection_health(collection_name: str):
        """检查Collection健康状态"""
        collection = Collection(collection_name)

        # 获取统计信息
        stats = collection.get_collection_stats()
        print(f"Total entities: {stats['row_count']}")

        # 检查索引状态
        indexes = collection.indexes
        for index in indexes:
            print(f"Index: {index.field_name}, Type: {index.params}")

        # 检查加载状态
        load_state = utility.load_state(collection_name)
        print(f"Load state: {load_state}")

    @staticmethod
    def compact_collection(collection_name: str):
        """压缩Collection（删除已标记删除的数据）"""
        collection = Collection(collection_name)
        collection.compact()
        print(f"Collection {collection_name} compacted")

    @staticmethod
    def get_query_performance(collection_name: str):
        """获取查询性能统计"""
        # 需要启用Milvus的性能监控
        # 这里是示例代码
        pass
```

### 8.5 分区策略（可选）

```python
# 按文档ID分区
def create_partitions_by_document(collection_name: str, document_ids: list):
    """
    为每个文档创建分区
    """
    collection = Collection(collection_name)

    for doc_id in document_ids:
        partition_name = f"doc_{doc_id}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
            print(f"Created partition: {partition_name}")

# 插入时指定分区
def insert_to_partition(collection, entity, document_id):
    """插入到指定分区"""
    partition_name = f"doc_{document_id}"
    partition = collection.partition(partition_name)
    partition.insert([entity])

# 查询时指定分区（提高性能）
def search_in_partition(collection, document_id, query_embedding):
    """在指定分区中检索"""
    partition_name = f"doc_{document_id}"
    partition = collection.partition(partition_name)
    results = partition.search(
        data=[query_embedding],
        anns_field="text_embedding",
        param={"metric_type": "IP"},
        limit=10
    )
    return results
```

---

## 9. 完整使用示例

```python
# ===== 初始化 =====
manager = MultimodalMilvusManager(host="localhost", port="19530")

# 创建Collection
collection = manager.create_multimodal_collection(
    collection_name="my_multimodal_rag",
    text_dim=768,
    image_dim=512
)

# 创建索引
manager.create_indexes("my_multimodal_rag")

# ===== 插入数据 =====
inserter = MultimodalDataInserter("my_multimodal_rag")

# 插入文本
text_id = inserter.insert_text_chunk(
    content="这是一段关于RAG的介绍...",
    text_embedding=[0.1] * 768,  # 实际应使用模型生成
    document_id="doc001",
    chunk_index=0,
    metadata={
        "title": "RAG技术白皮书",
        "category": "技术文档"
    }
)

# 插入图像
image_id = inserter.insert_image(
    caption="RAG系统架构图",
    image_url="https://example.com/rag_architecture.jpg",
    text_embedding=[0.2] * 768,
    image_embedding=[0.3] * 512,
    document_id="doc001",
    chunk_index=1,
    width=1920,
    height=1080,
    metadata={
        "image_metadata": {
            "format": "jpg",
            "caption": "图1：RAG架构"
        }
    }
)

# 插入表格
table_id = inserter.insert_table(
    table_content="| 模型 | 参数 |\\n| GPT-4 | 1.7T |",
    text_embedding=[0.4] * 768,
    document_id="doc001",
    chunk_index=2,
    rows=2,
    cols=2,
    metadata={
        "table_metadata": {
            "caption": "表1：模型对比"
        }
    }
)

# ===== 检索 =====
searcher = MultimodalSearcher("my_multimodal_rag")

# 文本检索
results = searcher.search_by_text(
    query_embedding=[0.15] * 768,
    top_k=10,
    content_types=["text", "table"]
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:100]}...")
    print("---")

# 混合检索
hybrid_results = searcher.hybrid_search(
    text_embedding=[0.15] * 768,
    image_embedding=[0.25] * 512,
    top_k=10,
    text_weight=0.6,
    image_weight=0.4
)

# ===== 数据管理 =====
data_manager = MultimodalDataManager("my_multimodal_rag")

# 查看统计
stats = data_manager.get_collection_stats()
print(stats)

# 删除文档
data_manager.delete_by_document_id("doc001")
```

---

## 10. 常见问题与解决方案

### Q1: 如何处理超长文本？

```python
# 方案1: 截断
max_length = 10000
truncated_content = content[:max_length]

# 方案2: 分块存储
def split_long_text(text, chunk_size=5000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# 方案3: 存储摘要+外部链接
entity = {
    "content": generate_summary(long_text),
    "content_url": "s3://bucket/full_text.txt"
}
```

### Q2: 如何优化图像向量存储？

```python
# 方案1: 降维
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
reduced_embedding = pca.fit_transform([image_embedding])[0]

# 方案2: 量化
def quantize_vector(vector, bits=8):
    """向量量化"""
    min_val, max_val = min(vector), max(vector)
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = [(int((v - min_val) * scale)) for v in vector]
    return quantized

# 方案3: 使用IVF_PQ索引（自动压缩）
```

### Q3: 如何实现增量更新？

```python
# Milvus不支持直接更新，需要删除后插入
def update_entity(entity_id, new_data):
    # 1. 查询现有数据
    old_entity = data_manager.get_by_id(entity_id)

    # 2. 删除旧数据
    data_manager.delete_by_ids([entity_id])

    # 3. 合并新数据
    old_entity.update(new_data)

    # 4. 插入更新后的数据
    inserter.collection.insert([old_entity])
```

### Q4: 如何备份和恢复？

```python
# 导出数据
def export_collection(collection_name, output_file):
    collection = Collection(collection_name)
    # 查询所有数据
    results = collection.query(
        expr="id != ''",
        output_fields=["*"]
    )

    # 保存到文件
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f)

# 导入数据
def import_collection(collection_name, input_file):
    import json
    with open(input_file, 'r') as f:
        entities = json.load(f)

    inserter = MultimodalDataInserter(collection_name)
    inserter.batch_insert(entities)
```

---

## 11. 总结

### 关键要点

1. **单Collection统一存储**适合大多数场景，简化管理
2. **合理设计metadata**，利用JSON字段存储灵活元数据
3. **选择合适的索引**，根据数据规模和性能需求
4. **批量操作**提高性能，避免频繁小批量插入
5. **零向量占位**处理可选向量字段
6. **监控和维护**定期检查Collection健康状态

### 性能建议

| 数据规模 | 索引类型 | 预期QPS | 内存需求 |
|---------|---------|---------|---------|
| <10万 | FLAT | 100 | 低 |
| 10万-100万 | IVF_FLAT | 500 | 中 |
| 100万-1000万 | HNSW | 1000+ | 高 |
| >1000万 | IVF_PQ | 2000+ | 中 |

---

**参考资料**:
- [Milvus官方文档](https://milvus.io/docs)
- [Pymilvus API参考](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)

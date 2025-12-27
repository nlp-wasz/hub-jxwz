# 多模态RAG系统 - API接口设计

## 1. API设计原则

### 1.1 保持现有API不变
所有现有文本RAG的API路由完全保持不变：
- `/upload` - 文本文档上传
- `/search` - 文本检索
- `/files/*` - 文件管理
- `/domains/*` - 领域管理

### 1.2 新增API命名规范
- 多模态相关API统一使用 `/multimodal` 前缀
- RESTful风格设计
- 支持领域隔离

---

## 2. 多模态文档管理API

### 2.1 上传多模态文档

**POST** `/api/v1/multimodal/upload`

**功能**: 上传PDF文档并进行多模态解析

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| file | File | 是 | PDF文件 |
| domain_name | string | 否 | 领域名称，默认"default" |
| parse_method | string | 否 | 解析方法: "paddleocr"/"mineru"，默认"paddleocr" |
| enable_ocr | boolean | 否 | 是否启用OCR，默认true |
| extract_images | boolean | 否 | 是否提取图片，默认true |
| generate_captions | boolean | 否 | 是否生成图片描述，默认false |
| vectorize_immediately | boolean | 否 | 是否立即向量化，默认false（异步） |

**请求示例**:
```bash
curl -X POST "http://localhost:8013/api/v1/multimodal/upload" \
  -F "file=@research_paper.pdf" \
  -F "domain_name=default" \
  -F "parse_method=paddleocr" \
  -F "extract_images=true" \
  -F "generate_captions=true"
```

**响应示例**:
```json
{
  "success": true,
  "message": "文档上传成功，正在异步处理",
  "data": {
    "doc_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "filename": "research_paper.pdf",
    "file_size": 2048576,
    "domain_name": "default",
    "parse_status": "parsing",
    "task_id": "task_xyz123",
    "estimated_time": "预计2-5分钟",
    "minio_paths": {
      "raw": "multimodal-raw/a1b2c3d4_research_paper.pdf"
    },
    "upload_time": "2025-11-13T10:00:00Z"
  }
}
```

### 2.2 查询文档解析状态

**GET** `/api/v1/multimodal/documents/{doc_id}/status`

**功能**: 查询文档处理进度

**响应示例**:
```json
{
  "success": true,
  "data": {
    "doc_id": "a1b2c3d4",
    "filename": "research_paper.pdf",
    "parse_status": "parsed",
    "parse_progress": 100,
    "vectorization_status": "processing",
    "vectorization_progress": 45,
    "milvus_sync_status": "pending",
    "statistics": {
      "total_pages": 10,
      "total_images": 5,
      "total_text_blocks": 120,
      "vectors_generated": {
        "bge": true,
        "clip_text": true,
        "clip_image": false
      }
    },
    "timestamps": {
      "uploaded": "2025-11-13T10:00:00Z",
      "parse_started": "2025-11-13T10:00:05Z",
      "parse_completed": "2025-11-13T10:02:30Z",
      "vectorization_started": "2025-11-13T10:02:35Z"
    }
  }
}
```

### 2.3 获取文档详情

**GET** `/api/v1/multimodal/documents/{doc_id}`

**查询参数**:
- `include_images`: boolean - 是否包含图片列表
- `include_text_blocks`: boolean - 是否包含文本块
- `include_relations`: boolean - 是否包含图文关系

**响应示例**:
```json
{
  "success": true,
  "data": {
    "doc_id": "a1b2c3d4",
    "filename": "research_paper.pdf",
    "domain": {
      "domain_id": "domain123",
      "domain_name": "default",
      "display_name": "默认领域"
    },
    "statistics": {
      "pages": 10,
      "images": 5,
      "text_blocks": 120,
      "chunks": 15
    },
    "images": [
      {
        "image_id": "img001",
        "page_number": 1,
        "image_path": "multimodal-images/a1b2c3d4/page_1/image_0.jpg",
        "caption": "销售趋势图",
        "image_type": "chart",
        "width": 800,
        "height": 600
      }
    ],
    "text_blocks": [
      {
        "block_id": "block001",
        "page_number": 1,
        "text_content": "第一章 引言",
        "block_type": "title",
        "char_length": 6
      }
    ],
    "relations": [
      {
        "relation_id": "rel001",
        "image_id": "img001",
        "text_block_id": "block005",
        "relation_type": "caption",
        "relation_score": 0.95
      }
    ]
  }
}
```

### 2.4 删除多模态文档

**DELETE** `/api/v1/multimodal/documents/{doc_id}`

**查询参数**:
- `force`: boolean - 强制删除（即使处理中）

**响应示例**:
```json
{
  "success": true,
  "message": "文档删除成功",
  "deleted_items": {
    "document": 1,
    "images": 5,
    "text_blocks": 120,
    "chunks": 15,
    "relations": 25,
    "milvus_vectors": 125,
    "minio_files": 6
  }
}
```

---

## 3. 多模态检索API

### 3.1 统一多模态检索接口

**POST** `/api/v1/multimodal/search`

**功能**: 支持文本、图像、文本+图像的多模态检索

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| query_text | string | 条件必填 | 查询文本 |
| query_image | File | 条件必填 | 查询图片（query_text和query_image至少一个） |
| search_mode | string | 是 | 检索模式详见下表 |
| domain_name | string | 否 | 领域名称 |
| cross_domain | boolean | 否 | 是否跨域检索，默认false |
| top_k | integer | 否 | 返回结果数，默认10 |
| return_images | boolean | 否 | 是否返回图片，默认true |
| return_text | boolean | 否 | 是否返回文本，默认true |
| filters | object | 否 | 过滤条件 |

**检索模式**:

| search_mode | 输入 | 检索目标 | 向量库 |
|------------|------|---------|--------|
| text2text | 文本 | 文本块 | mm_text_bge |
| text2image | 文本 | 图片 | mm_image_clip |
| image2text | 图片 | 文本块 | mm_text_clip |
| image2image | 图片 | 图片 | mm_image_clip |
| multimodal | 文本+图片 | 图文混合 | 多集合融合 |

**请求示例（text2image）**:
```bash
curl -X POST "http://localhost:8013/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "销售趋势图",
    "search_mode": "text2image",
    "domain_name": "default",
    "top_k": 5,
    "return_images": true
  }'
```

**响应示例**:
```json
{
  "success": true,
  "query": {
    "text": "销售趋势图",
    "mode": "text2image"
  },
  "results": [
    {
      "rank": 1,
      "score": 0.8756,
      "type": "image",
      "image": {
        "image_id": "img001",
        "doc_id": "a1b2c3d4",
        "filename": "research_paper.pdf",
        "page_number": 3,
        "image_path": "multimodal-images/a1b2c3d4/page_3/image_0.jpg",
        "image_url": "http://minio:9000/multimodal-images/a1b2c3d4/page_3/image_0.jpg",
        "caption": "2024年销售趋势分析图",
        "image_type": "chart",
        "width": 800,
        "height": 600
      },
      "context": {
        "related_text": [
          {
            "text": "如图所示，销售额在Q2达到峰值...",
            "block_id": "block025",
            "relation_type": "caption"
          }
        ]
      }
    }
  ],
  "stats": {
    "total_candidates": 50,
    "returned": 5,
    "search_time_ms": 45
  }
}
```

### 3.2 图文混合检索（高级）

**POST** `/api/v1/multimodal/search/hybrid`

**请求参数**:
```json
{
  "query_text": "分析产品销售趋势",
  "query_image": "base64_encoded_image",
  "search_config": {
    "text_weight": 0.6,
    "image_weight": 0.4,
    "fusion_method": "weighted_sum",
    "enable_rerank": true
  },
  "filters": {
    "domain_name": "sales",
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "document_types": ["report", "presentation"]
  },
  "top_k": 10
}
```

**响应示例**:
```json
{
  "success": true,
  "results": [
    {
      "rank": 1,
      "score": 0.9123,
      "type": "multimodal_chunk",
      "chunk": {
        "chunk_id": "chunk001",
        "doc_id": "a1b2c3d4",
        "text_content": "根据销售数据分析...",
        "images": [
          {
            "image_id": "img001",
            "image_url": "...",
            "caption": "销售趋势图"
          }
        ],
        "page_range": "3-4"
      },
      "relevance": {
        "text_score": 0.85,
        "image_score": 0.92,
        "combined_score": 0.9123
      }
    }
  ]
}
```

### 3.3 两种检索接口的详细对比

#### 统一多模态检索 vs 图文混合检索

| 对比维度 | 统一多模态检索 `/multimodal/search` | 图文混合检索 `/multimodal/search/hybrid` |
|---------|----------------------------------|--------------------------------------|
| **使用场景** | 单一模态检索 | 图文联合检索 |
| **输入方式** | 文本 **OR** 图片 | 文本 **AND** 图片 |
| **检索模式** | 通过`search_mode`选择5种模式之一 | 固定为multimodal混合模式 |
| **配置复杂度** | 简单（仅search_mode参数） | 复杂（权重、融合算法、重排序） |
| **权重控制** | ❌ 不支持 | ✅ 支持text_weight、image_weight |
| **融合算法** | 默认算法 | 可选weighted_sum/RRF/其他 |
| **重排序** | ❌ 不支持 | ✅ 可选DashScope重排序 |
| **过滤条件** | 基础过滤 | 高级过滤（时间范围、文档类型等） |
| **适用对象** | 一般用户、简单查询 | 高级用户、复杂查询 |
| **典型用例** | "找销售相关的图表" | "结合这张图和'销售'关键词找最相关的内容" |

#### 五种检索模式详解

**1. text2text - 纯文本检索**
- **向量模型**: BGE (BAAI/bge-small-zh-v1.5)
- **向量库**: mm_text_bge
- **应用场景**: 传统文本语义检索
- **示例**: 输入"销售数据分析"，检索相关文本段落

**2. text2image - 文本搜图（跨模态）**
- **向量模型**: CLIP文本编码器
- **向量库**: mm_image_clip
- **应用场景**: 用文字描述查找图片
- **示例**: 输入"产品架构图"，检索出架构示意图

**3. image2text - 图搜文（跨模态）**
- **向量模型**: CLIP图像编码器
- **向量库**: mm_text_clip
- **应用场景**: 用图片查找相关文本说明
- **示例**: 上传一张图表，检索出相关的文字描述

**4. image2image - 以图搜图**
- **向量模型**: CLIP图像编码器
- **向量库**: mm_image_clip
- **应用场景**: 查找相似图片
- **示例**: 上传一张柱状图，查找其他类似的柱状图

**5. multimodal - 图文混合检索**
- **向量模型**: CLIP文本+图像双编码
- **向量库**: 多集合融合查询
- **应用场景**: 综合图文信息查询
- **示例**: 同时输入文本"销售"和一张趋势图，综合检索

#### 使用建议

```python
# 场景1: 只有文本查询 → 使用统一接口 + text2text模式
{
  "query_text": "产品销售报告",
  "search_mode": "text2text"
}

# 场景2: 想找图片 → 使用统一接口 + text2image模式
{
  "query_text": "销售趋势图",
  "search_mode": "text2image"
}

# 场景3: 有图片和文本，需要精确控制权重 → 使用混合接口
{
  "query_text": "销售分析",
  "query_image": "base64_image...",
  "search_config": {
    "text_weight": 0.7,  # 更重视文本
    "image_weight": 0.3
  }
}
```

---

## 4. 图文问答API

### 4.1 基于检索的问答

**POST** `/api/v1/multimodal/qa`

**功能**: 基于Qwen-VL的图文理解问答

**请求参数**:
```json
{
  "question": "这个图表显示了什么趋势？",
  "context_config": {
    "search_mode": "multimodal",
    "domain_name": "sales",
    "top_k": 3,
    "include_images": true,
    "max_context_length": 2000
  },
  "answer_config": {
    "model": "qwen-vl-plus",
    "max_tokens": 500,
    "temperature": 0.7,
    "return_sources": true
  }
}
```

**响应示例**:
```json
{
  "success": true,
  "question": "这个图表显示了什么趋势？",
  "answer": "根据图表分析，2024年销售额呈现明显的上升趋势，Q2达到峰值后略有回落，整体增长率约35%。",
  "confidence": 0.89,
  "sources": [
    {
      "type": "image",
      "image_id": "img001",
      "doc_id": "a1b2c3d4",
      "filename": "sales_report.pdf",
      "page_number": 3,
      "image_url": "...",
      "relevance_score": 0.95
    },
    {
      "type": "text",
      "block_id": "block025",
      "text": "销售数据显示，Q2销售额达到...",
      "relevance_score": 0.87
    }
  ],
  "metadata": {
    "model": "qwen-vl-plus",
    "tokens_used": 345,
    "processing_time_ms": 1200
  }
}
```

### 4.2 直接图片问答（无检索）

**POST** `/api/v1/multimodal/qa/direct`

**功能**: 直接对上传图片进行问答

**请求参数**:
- `image`: File - 图片文件
- `question`: string - 问题
- `model`: string - 模型名称

**响应示例**:
```json
{
  "success": true,
  "question": "这张图片展示的是什么？",
  "answer": "这是一张柱状图，展示了2024年各季度的销售额对比...",
  "image_analysis": {
    "image_type": "chart",
    "detected_objects": ["bar_chart", "axis", "legend"],
    "ocr_text": ["Q1", "Q2", "Q3", "Q4", "销售额"]
  }
}
```

### 4.3 两种问答接口的详细对比

#### 基于检索的问答 vs 直接图片问答

| 对比维度 | 基于检索的问答 `/multimodal/qa` | 直接图片问答 `/multimodal/qa/direct` |
|---------|-------------------------------|----------------------------------|
| **是否检索知识库** | ✅ 需要先检索知识库 | ❌ 不检索，直接理解 |
| **图片来源** | 从已入库的文档中检索 | 用户临时上传 |
| **上下文信息** | 包含检索到的图文上下文 | 仅当前图片 |
| **答案准确性** | 基于知识库，准确度高 | 基于视觉理解，可能泛化 |
| **处理流程** | 检索 → 构建上下文 → Qwen-VL理解 | 直接 Qwen-VL理解 |
| **响应速度** | 较慢（含检索时间） | 较快（无检索） |
| **信息来源追溯** | ✅ 返回sources字段 | ❌ 无来源信息 |
| **适用场景** | 针对已入库文档的问答 | 临时图片快速理解 |

#### 使用场景示例

**场景1: 基于检索的问答（企业内部知识库）**
```json
// 问题：我们公司Q2的销售趋势如何？
// → 先检索知识库中的销售文档和图表
// → Qwen-VL基于检索结果回答
{
  "question": "我们公司Q2的销售趋势如何？",
  "context_config": {
    "search_mode": "multimodal",
    "domain_name": "sales",
    "top_k": 3
  }
}

// 响应包含：
// - answer: "根据销售报告，Q2销售额增长35%..."
// - sources: [销售报告.pdf 第3页图表, 第5页文字描述]
```

**场景2: 直接图片问答（临时文件）**
```json
// 用户临时上传一张发票图片
// 问题：这张发票的金额是多少？
// → 直接让Qwen-VL理解图片，无需入库

POST /api/v1/multimodal/qa/direct
Content-Type: multipart/form-data

image: [发票图片文件]
question: "这张发票的金额是多少？"

// 响应：
{
  "answer": "发票金额为 1,234.56 元",
  "image_analysis": {
    "image_type": "invoice",
    "ocr_text": ["金额", "1,234.56", "元"]
  }
}
```

**场景3: 基于检索问答的典型应用**
- 企业知识库问答
- 研究论文理解
- 产品文档查询
- 历史数据分析

**场景4: 直接问答的典型应用**
- 临时文件快速识别
- 图表数据提取
- 发票/票据识别
- 图片内容描述

---

## 5. 图片管理API

### 5.1 获取图片列表

**GET** `/api/v1/multimodal/images`

**查询参数**:
- `doc_id`: string - 文档ID
- `page_number`: integer - 页码
- `image_type`: string - 图片类型筛选
- `limit`: integer - 返回数量
- `offset`: integer - 偏移量

**响应示例**:
```json
{
  "success": true,
  "total": 25,
  "images": [
    {
      "image_id": "img001",
      "doc_id": "a1b2c3d4",
      "page_number": 1,
      "image_url": "...",
      "thumbnail_url": "...",
      "caption": "产品架构图",
      "image_type": "diagram",
      "width": 1024,
      "height": 768
    }
  ]
}
```

### 5.2 获取图片详情

**GET** `/api/v1/multimodal/images/{image_id}`

**响应包含**:
- 图片基本信息
- 关联的文本块
- 向量化状态
- 相似图片推荐

---

## 6. 批量操作API

### 6.1 批量文档上传

**POST** `/api/v1/multimodal/batch/upload`

**请求**: multipart/form-data，多个文件

**响应**:
```json
{
  "success": true,
  "total": 10,
  "succeeded": 9,
  "failed": 1,
  "results": [
    {
      "filename": "doc1.pdf",
      "status": "success",
      "doc_id": "xxx"
    },
    {
      "filename": "doc2.pdf",
      "status": "failed",
      "error": "文件格式不支持"
    }
  ]
}
```

### 6.2 批量向量化

**POST** `/api/v1/multimodal/batch/vectorize`

**请求**:
```json
{
  "doc_ids": ["doc1", "doc2", "doc3"],
  "vector_types": ["bge", "clip_text", "clip_image"],
  "force_regenerate": false
}
```

---

## 7. 统计和监控API

### 7.1 系统统计

**GET** `/api/v1/multimodal/stats`

**响应**:
```json
{
  "success": true,
  "stats": {
    "total_documents": 150,
    "total_images": 750,
    "total_text_blocks": 18000,
    "total_vectors": {
      "bge": 18000,
      "clip_text": 18000,
      "clip_image": 750
    },
    "storage": {
      "minio_total_mb": 5120,
      "mysql_size_mb": 256,
      "milvus_size_mb": 1024
    },
    "by_domain": {
      "default": {
        "documents": 100,
        "images": 500
      }
    }
  }
}
```

### 7.2 处理队列状态

**GET** `/api/v1/multimodal/queue/status`

**响应**:
```json
{
  "success": true,
  "queue_status": {
    "parsing": {
      "pending": 5,
      "processing": 2,
      "completed": 143,
      "failed": 1
    },
    "vectorization": {
      "pending": 3,
      "processing": 1,
      "completed": 145,
      "failed": 0
    }
  }
}
```

---

## 8. 错误处理

### 8.1 错误码定义

| 错误码 | 说明 |
|-------|------|
| 40001 | 文件格式不支持 |
| 40002 | 文件大小超限 |
| 40003 | 领域不存在 |
| 40004 | 文档不存在 |
| 40005 | 图片不存在 |
| 50001 | 解析失败 |
| 50002 | 向量化失败 |
| 50003 | Milvus同步失败 |
| 50004 | 模型服务不可用 |

### 8.2 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": 40001,
    "message": "不支持的文件格式",
    "details": "仅支持PDF文件",
    "timestamp": "2025-11-13T10:00:00Z"
  }
}
```

---

## 9. 限流和配额

### 9.1 速率限制

- 文档上传: 10次/分钟
- 检索请求: 100次/分钟
- 问答请求: 20次/分钟

### 9.2 配额限制

- 单文件大小: 50MB
- 单文档最大页数: 500页
- 单次检索top_k: 最大100

---

## 10. API认证（预留）

```python
# 请求头
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "X-Domain": "default"
}
```

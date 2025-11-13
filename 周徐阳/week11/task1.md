# 多模态RAG系统接口设计文档

## 1. 数据管理接口

### 1.1 上传文档
**接口路径**: `POST /api/v1/documents/upload`

**功能描述**: 上传多模态文档数据，支持文本、图片、PDF等格式

**请求参数**:
```json
{
  "document_type": "text|image|pdf|mixed",
  "file": "base64编码的文件内容 或 文件URL",
  "metadata": {
    "title": "文档标题",
    "source": "数据来源",
    "tags": ["标签1", "标签2"],
    "category": "分类",
    "author": "作者",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "processing_options": {
    "extract_images": true,
    "chunk_size": 512,
    "overlap": 50,
    "enable_ocr": true,
    "generate_summary": true
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "文档上传成功",
  "data": {
    "document_id": "doc_123456789",
    "status": "processing|completed|failed",
    "chunks_count": 15,
    "images_count": 3,
    "processing_time": 2.5,
    "index_status": "indexed",
    "embeddings_generated": true
  }
}
```

---

### 1.2 批量上传文档
**接口路径**: `POST /api/v1/documents/batch-upload`

**功能描述**: 批量上传多个文档

**请求参数**:
```json
{
  "documents": [
    {
      "document_type": "text|image|pdf|mixed",
      "file": "文件内容或URL",
      "metadata": {}
    }
  ],
  "processing_options": {}
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "批量上传完成",
  "data": {
    "total": 10,
    "success": 8,
    "failed": 2,
    "document_ids": ["doc_123", "doc_124"],
    "failed_items": [
      {
        "index": 5,
        "error": "文件格式不支持"
      }
    ]
  }
}
```

---

### 1.3 删除文档
**接口路径**: `DELETE /api/v1/documents/{document_id}`

**功能描述**: 删除指定文档及其向量索引

**请求参数**:
- Path参数: `document_id` (必填)
- Query参数:
  - `delete_related`: boolean (是否删除相关chunks，默认true)

**返回结果**:
```json
{
  "code": 200,
  "message": "文档删除成功",
  "data": {
    "document_id": "doc_123456789",
    "deleted_chunks": 15,
    "deleted_embeddings": 15
  }
}
```

---

### 1.4 查询文档列表
**接口路径**: `GET /api/v1/documents`

**功能描述**: 分页查询文档列表，支持筛选

**请求参数**:
- Query参数:
  - `page`: int (页码，默认1)
  - `page_size`: int (每页数量，默认20)
  - `document_type`: string (文档类型筛选)
  - `category`: string (分类筛选)
  - `tags`: string (标签筛选，逗号分隔)
  - `keyword`: string (关键词搜索)
  - `sort_by`: string (排序字段: created_at|updated_at|title)
  - `order`: string (排序方式: asc|desc)

**返回结果**:
```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "total": 100,
    "page": 1,
    "page_size": 20,
    "documents": [
      {
        "document_id": "doc_123456789",
        "document_type": "mixed",
        "metadata": {
          "title": "示例文档",
          "source": "upload",
          "tags": ["AI", "RAG"],
          "category": "技术文档"
        },
        "chunks_count": 15,
        "images_count": 3,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "status": "completed"
      }
    ]
  }
}
```

---

### 1.5 获取文档详情
**接口路径**: `GET /api/v1/documents/{document_id}`

**功能描述**: 获取文档详细信息

**请求参数**:
- Path参数: `document_id` (必填)

**返回结果**:
```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "document_id": "doc_123456789",
    "document_type": "mixed",
    "metadata": {
      "title": "示例文档",
      "source": "upload",
      "tags": ["AI", "RAG"],
      "category": "技术文档",
      "author": "张三",
      "created_at": "2024-01-01T00:00:00Z"
    },
    "content_preview": "文档内容预览...",
    "chunks": [
      {
        "chunk_id": "chunk_001",
        "content": "文本块内容",
        "chunk_index": 0,
        "token_count": 150
      }
    ],
    "images": [
      {
        "image_id": "img_001",
        "url": "https://example.com/image.jpg",
        "caption": "图片描述",
        "position": 0
      }
    ],
    "statistics": {
      "total_chunks": 15,
      "total_images": 3,
      "total_tokens": 2048,
      "file_size": "2.5MB"
    }
  }
}
```

---

### 1.6 更新文档元数据
**接口路径**: `PUT /api/v1/documents/{document_id}/metadata`

**功能描述**: 更新文档的元数据信息

**请求参数**:
```json
{
  "metadata": {
    "title": "新标题",
    "tags": ["标签1", "标签2"],
    "category": "新分类"
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "更新成功",
  "data": {
    "document_id": "doc_123456789",
    "updated_fields": ["title", "tags", "category"],
    "updated_at": "2024-01-02T00:00:00Z"
  }
}
```

---

## 2. 多模态检索接口

### 2.1 文本检索
**接口路径**: `POST /api/v1/search/text`

**功能描述**: 基于文本查询进行语义检索

**请求参数**:
```json
{
  "query": "检索查询文本",
  "top_k": 10,
  "filters": {
    "document_type": ["text", "mixed"],
    "category": "技术文档",
    "tags": ["AI"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  },
  "search_options": {
    "similarity_threshold": 0.7,
    "rerank": true,
    "rerank_model": "bge-reranker-v2",
    "include_metadata": true,
    "include_content": true
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "检索成功",
  "data": {
    "query": "检索查询文本",
    "total_results": 10,
    "search_time": 0.15,
    "results": [
      {
        "chunk_id": "chunk_001",
        "document_id": "doc_123456789",
        "content": "匹配的文本内容...",
        "similarity_score": 0.95,
        "rerank_score": 0.92,
        "metadata": {
          "title": "文档标题",
          "source": "upload",
          "category": "技术文档"
        },
        "highlight": {
          "content": "高亮显示的<em>关键词</em>内容"
        },
        "position": {
          "chunk_index": 5,
          "page_number": 2
        }
      }
    ]
  }
}
```

---

### 2.2 图像检索
**接口路径**: `POST /api/v1/search/image`

**功能描述**: 基于图像或图像描述进行检索

**请求参数**:
```json
{
  "query_type": "image|text",
  "query_image": "base64编码的图片 或 图片URL",
  "query_text": "图片描述文本（当query_type为text时使用）",
  "top_k": 10,
  "filters": {
    "document_type": ["image", "mixed"],
    "category": "产品图册"
  },
  "search_options": {
    "similarity_threshold": 0.7,
    "include_similar_text": true,
    "return_image_url": true
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "检索成功",
  "data": {
    "query_type": "image",
    "total_results": 10,
    "search_time": 0.25,
    "results": [
      {
        "image_id": "img_001",
        "document_id": "doc_123456789",
        "image_url": "https://example.com/image.jpg",
        "thumbnail_url": "https://example.com/thumbnail.jpg",
        "similarity_score": 0.89,
        "caption": "图片描述或OCR文本",
        "metadata": {
          "title": "文档标题",
          "position": 3,
          "size": "1024x768"
        },
        "related_text": {
          "before": "图片前的文本内容",
          "after": "图片后的文本内容"
        }
      }
    ]
  }
}
```

---

### 2.3 混合多模态检索
**接口路径**: `POST /api/v1/search/multimodal`

**功能描述**: 同时使用文本和图像进行混合检索

**请求参数**:
```json
{
  "query_text": "文本查询",
  "query_image": "base64编码的图片（可选）",
  "modality_weights": {
    "text": 0.6,
    "image": 0.4
  },
  "top_k": 10,
  "return_modalities": ["text", "image", "mixed"],
  "filters": {
    "category": "产品手册"
  },
  "search_options": {
    "fusion_strategy": "weighted_sum|rrf|linear_combination",
    "similarity_threshold": 0.7,
    "rerank": true,
    "group_by_document": true
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "检索成功",
  "data": {
    "query_text": "文本查询",
    "query_image_provided": true,
    "total_results": 10,
    "search_time": 0.35,
    "results": [
      {
        "result_id": "result_001",
        "result_type": "mixed",
        "document_id": "doc_123456789",
        "content": {
          "text": "文本内容...",
          "images": [
            {
              "image_url": "https://example.com/image.jpg",
              "caption": "图片描述"
            }
          ]
        },
        "scores": {
          "text_similarity": 0.92,
          "image_similarity": 0.85,
          "combined_score": 0.89,
          "rerank_score": 0.91
        },
        "metadata": {
          "title": "文档标题",
          "category": "产品手册"
        }
      }
    ],
    "aggregation": {
      "by_document": [
        {
          "document_id": "doc_123456789",
          "match_count": 5,
          "avg_score": 0.88
        }
      ]
    }
  }
}
```

---

### 2.4 相似内容推荐
**接口路径**: `POST /api/v1/search/similar`

**功能描述**: 基于指定内容推荐相似内容

**请求参数**:
```json
{
  "reference_type": "document|chunk|image",
  "reference_id": "doc_123456789",
  "top_k": 5,
  "exclude_self": true,
  "filters": {}
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "推荐成功",
  "data": {
    "reference_id": "doc_123456789",
    "total_results": 5,
    "results": [
      {
        "item_id": "doc_987654321",
        "item_type": "document",
        "similarity_score": 0.88,
        "metadata": {
          "title": "相似文档标题"
        }
      }
    ]
  }
}
```

---

## 3. 多模态问答接口

### 3.1 单轮问答
**接口路径**: `POST /api/v1/qa/ask`

**功能描述**: 基于RAG的单轮问答

**请求参数**:
```json
{
  "question": "用户问题",
  "question_image": "base64编码的图片（可选）",
  "context_images": ["图片URL1", "图片URL2"],
  "retrieval_config": {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "enable_rerank": true,
    "search_mode": "text|image|multimodal"
  },
  "generation_config": {
    "model": "gpt-4-vision|claude-3|qwen-vl",
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": false
  },
  "filters": {
    "document_ids": ["doc_123", "doc_456"],
    "category": "技术文档"
  },
  "options": {
    "include_sources": true,
    "include_images": true,
    "language": "zh"
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "问答成功",
  "data": {
    "question": "用户问题",
    "answer": "基于检索内容生成的答案...",
    "confidence_score": 0.92,
    "response_time": 1.5,
    "sources": [
      {
        "source_id": "chunk_001",
        "document_id": "doc_123456789",
        "content": "引用的原文内容...",
        "relevance_score": 0.95,
        "metadata": {
          "title": "来源文档标题",
          "page": 5
        }
      }
    ],
    "retrieved_images": [
      {
        "image_id": "img_001",
        "image_url": "https://example.com/image.jpg",
        "caption": "图片描述",
        "relevance_score": 0.88
      }
    ],
    "reasoning": {
      "retrieval_summary": "检索到5个相关片段",
      "answer_based_on": ["chunk_001", "chunk_003"]
    }
  }
}
```

---

### 3.2 流式问答
**接口路径**: `POST /api/v1/qa/ask-stream`

**功能描述**: 支持流式返回的问答接口

**请求参数**: 同 3.1，但 `generation_config.stream` 必须为 true

**返回结果** (SSE格式):
```
data: {"type": "retrieval_start", "data": {"status": "searching"}}

data: {"type": "retrieval_complete", "data": {"retrieved_count": 5}}

data: {"type": "generation_start", "data": {"model": "gpt-4-vision"}}

data: {"type": "content_chunk", "data": {"text": "答案内容片段1..."}}

data: {"type": "content_chunk", "data": {"text": "答案内容片段2..."}}

data: {"type": "sources", "data": {"sources": [...]}}

data: {"type": "complete", "data": {"total_tokens": 500}}
```

---

### 3.3 多轮对话问答
**接口路径**: `POST /api/v1/qa/chat`

**功能描述**: 支持上下文的多轮对话问答

**请求参数**:
```json
{
  "session_id": "session_123456",
  "question": "当前问题",
  "question_image": "base64编码的图片（可选）",
  "history": [
    {
      "role": "user",
      "content": "之前的问题",
      "images": []
    },
    {
      "role": "assistant",
      "content": "之前的回答",
      "sources": []
    }
  ],
  "retrieval_config": {
    "use_history_context": true,
    "history_weight": 0.3,
    "top_k": 5
  },
  "generation_config": {
    "model": "gpt-4-vision",
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "options": {
    "include_sources": true,
    "save_session": true
  }
}
```

**返回结果**:
```json
{
  "code": 200,
  "message": "问答成功",
  "data": {
    "session_id": "session_123456",
    "question": "当前问题",
    "answer": "基于上下文和检索内容的答案...",
    "confidence_score": 0.91,
    "response_time": 2.1,
    "sources": [],
    "retrieved_images": [],
    "context_used": {
      "history_turns": 2,
      "retrieved_chunks": 5
    },
    "follow_up_suggestions": [
      "可以进一步询问的问题1",
      "可以进一步询问的问题2"
    ]
  }
}
```

---

### 3.4 获取对话历史
**接口路径**: `GET /api/v1/qa/sessions/{session_id}`

**功能描述**: 获取对话会话历史

**请求参数**:
- Path参数: `session_id` (必填)

**返回结果**:
```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "session_id": "session_123456",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T01:00:00Z",
    "message_count": 10,
    "messages": [
      {
        "message_id": "msg_001",
        "role": "user",
        "content": "问题内容",
        "images": [],
        "timestamp": "2024-01-01T00:00:00Z"
      },
      {
        "message_id": "msg_002",
        "role": "assistant",
        "content": "回答内容",
        "sources": [],
        "timestamp": "2024-01-01T00:00:05Z"
      }
    ]
  }
}
```

---

### 3.5 删除对话会话
**接口路径**: `DELETE /api/v1/qa/sessions/{session_id}`

**功能描述**: 删除指定对话会话

**请求参数**:
- Path参数: `session_id` (必填)

**返回结果**:
```json
{
  "code": 200,
  "message": "会话删除成功",
  "data": {
    "session_id": "session_123456",
    "deleted_messages": 10
  }
}
```

---

## 4. 通用响应格式

### 4.1 成功响应
```json
{
  "code": 200,
  "message": "操作成功",
  "data": {}
}
```

### 4.2 错误响应
```json
{
  "code": 400|401|403|404|500,
  "message": "错误描述信息",
  "error": {
    "error_code": "INVALID_PARAMETER",
    "error_detail": "详细错误信息",
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### 4.3 常见错误码
| 错误码 | HTTP状态码 | 说明 |
|--------|-----------|------|
| SUCCESS | 200 | 请求成功 |
| INVALID_PARAMETER | 400 | 参数错误 |
| UNAUTHORIZED | 401 | 未授权 |
| FORBIDDEN | 403 | 禁止访问 |
| NOT_FOUND | 404 | 资源不存在 |
| INTERNAL_ERROR | 500 | 服务器内部错误 |
| SERVICE_UNAVAILABLE | 503 | 服务暂不可用 |
| RATE_LIMIT_EXCEEDED | 429 | 请求频率超限 |
| FILE_TOO_LARGE | 413 | 文件过大 |
| UNSUPPORTED_FORMAT | 415 | 不支持的文件格式 |

---

## 5. 认证与鉴权

所有接口需要在Header中携带认证信息：

```
Authorization: Bearer {access_token}
Content-Type: application/json
X-Request-ID: {唯一请求ID}
```

---

## 6. 限流规则

| 接口类型 | 限流规则 |
|---------|---------|
| 文档上传 | 100次/小时 |
| 数据查询 | 1000次/小时 |
| 检索接口 | 500次/小时 |
| 问答接口 | 200次/小时 |

---

## 7. 性能指标

| 接口 | 预期响应时间 | 超时时间 |
|-----|------------|---------|
| 文档上传 | < 5s | 30s |
| 文本检索 | < 200ms | 5s |
| 图像检索 | < 500ms | 10s |
| 混合检索 | < 800ms | 15s |
| 单轮问答 | < 3s | 30s |
| 流式问答 | 首字延迟 < 1s | 60s |

---

## 8. 使用示例

### Python示例
```python
import requests
import base64

# 1. 上传文档
def upload_document(file_path, api_key):
    with open(file_path, 'rb') as f:
        file_content = base64.b64encode(f.read()).decode()

    url = "http://api.example.com/api/v1/documents/upload"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "document_type": "pdf",
        "file": file_content,
        "metadata": {
            "title": "示例文档",
            "tags": ["AI", "RAG"]
        }
    }

    response = requests.post(url, json=data, headers=headers)
    return response.json()

# 2. 多模态检索
def multimodal_search(query_text, query_image_path, api_key):
    with open(query_image_path, 'rb') as f:
        image_content = base64.b64encode(f.read()).decode()

    url = "http://api.example.com/api/v1/search/multimodal"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "query_text": query_text,
        "query_image": image_content,
        "top_k": 10,
        "modality_weights": {
            "text": 0.6,
            "image": 0.4
        }
    }

    response = requests.post(url, json=data, headers=headers)
    return response.json()

# 3. 多模态问答
def multimodal_qa(question, api_key, question_image_path=None):
    url = "http://api.example.com/api/v1/qa/ask"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "question": question,
        "retrieval_config": {
            "top_k": 5,
            "enable_rerank": True
        },
        "generation_config": {
            "model": "gpt-4-vision",
            "temperature": 0.7
        },
        "options": {
            "include_sources": True
        }
    }

    if question_image_path:
        with open(question_image_path, 'rb') as f:
            data["question_image"] = base64.b64encode(f.read()).decode()

    response = requests.post(url, json=data, headers=headers)
    return response.json()
```

### cURL示例
```bash
# 上传文档
curl -X POST "http://api.example.com/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "text",
    "file": "文件内容",
    "metadata": {
      "title": "示例文档"
    }
  }'

# 文本检索
curl -X POST "http://api.example.com/api/v1/search/text" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "检索查询",
    "top_k": 10
  }'

# 问答
curl -X POST "http://api.example.com/api/v1/qa/ask" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "你的问题",
    "retrieval_config": {
      "top_k": 5
    }
  }'
```

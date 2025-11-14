我来为您设计一个完整的多模态RAG系统接口方案。以下是一个基于RESTful风格的接口设计：

## 1. 数据管理接口

### 1.1 上传多模态文档
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

**请求参数：**
```json
{
  "files": ["文件数组（支持图片、PDF、视频、音频等）"],
  "metadata": {
    "title": "文档标题",
    "description": "文档描述",
    "tags": ["标签1", "标签2"],
    "category": "文档分类",
    "visibility": "public/private"
  },
  "chunk_config": {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "modality_specific": {
      "image": {"extract_text": true, "generate_caption": true},
      "video": {"extract_frames": true, "frame_interval": 5},
      "audio": {"transcribe": true}
    }
  }
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456",
    "chunks": [
      {
        "chunk_id": "chunk_001",
        "modality": "text/image/video/audio",
        "content_preview": "内容预览...",
        "embedding_status": "success"
      }
    ],
    "processing_summary": {
      "total_chunks": 15,
      "successful": 15,
      "failed": 0
    }
  }
}
```

### 1.2 查询文档列表
```http
GET /api/v1/documents?page=1&page_size=20&category=research&search=关键词
```

**响应：**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "document_id": "doc_123456",
        "title": "文档标题",
        "description": "文档描述",
        "modalities": ["text", "image"],
        "created_at": "2024-01-15T10:30:00Z",
        "chunk_count": 15,
        "file_size": "2.5MB"
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 5,
      "total_documents": 95,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

### 1.3 删除文档
```http
DELETE /api/v1/documents/{document_id}
```

**响应：**
```json
{
  "success": true,
  "message": "文档删除成功",
  "data": {
    "deleted_chunks": 15,
    "deleted_embeddings": 15
  }
}
```

## 2. 多模态检索接口

### 2.1 混合模态检索
```http
POST /api/v1/retrieve/multi-modal
Content-Type: application/json
```

**请求参数：**
```json
{
  "query": {
    "text": "检索文本查询",
    "image": "base64编码的图片（可选）",
    "audio": "base64编码的音频（可选）"
  },
  "modality_weights": {
    "text": 0.6,
    "image": 0.3,
    "audio": 0.1
  },
  "retrieval_config": {
    "top_k": 10,
    "score_threshold": 0.7,
    "modality_filters": ["text", "image"],
    "rerank": true
  },
  "document_filters": {
    "categories": ["research", "manual"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "tags": ["重要", "技术"]
  }
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "chunk_id": "chunk_001",
        "document_id": "doc_123456",
        "modality": "text",
        "content": "检索到的文本内容...",
        "metadata": {
          "document_title": "源文档标题",
          "page_number": 5,
          "section": "章节标题"
        },
        "score": 0.89,
        "modality_scores": {
          "text": 0.92,
          "image": 0.45,
          "overall": 0.89
        }
      },
      {
        "chunk_id": "chunk_002",
        "document_id": "doc_123456",
        "modality": "image",
        "content": "base64编码的图片或图片URL",
        "caption": "图片的自动生成描述",
        "metadata": {
          "document_title": "源文档标题",
          "image_size": "800x600"
        },
        "score": 0.76,
        "modality_scores": {
          "text": 0.35,
          "image": 0.85,
          "overall": 0.76
        }
      }
    ],
    "retrieval_metrics": {
      "total_candidates": 150,
      "retrieval_time": "0.45s",
      "modality_breakdown": {
        "text": 6,
        "image": 3,
        "audio": 1
      }
    }
  }
}
```

### 2.2 跨模态检索
```http
POST /api/v1/retrieve/cross-modal
Content-Type: multipart/form-data
```

**请求参数：**
- `query_image` (file): 用图片检索相关内容
- `query_audio` (file): 用音频检索相关内容
- `target_modality` (string): 要检索的目标模态

**响应：** 与混合模态检索类似，但专注于跨模态检索结果。

## 3. 多模态问答接口

### 3.1 多轮对话问答
```http
POST /api/v1/chat/multi-turn
Content-Type: application/json
```

**请求参数：**
```json
{
  "session_id": "session_123456",
  "messages": [
    {
      "role": "user",
      "content": {
        "text": "用户问题文本",
        "image": "base64编码的图片（可选）",
        "audio": "base64编码的音频（可选）"
      },
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "conversation_history": [
    {
      "role": "user",
      "content": "之前的问题...",
      "timestamp": "2024-01-15T10:25:00Z"
    },
    {
      "role": "assistant", 
      "content": "之前的回答...",
      "sources": ["chunk_001", "chunk_002"],
      "timestamp": "2024-01-15T10:26:00Z"
    }
  ],
  "qa_config": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "include_sources": true,
    "include_images": true,
    "reasoning_depth": "detailed"
  },
  "retrieval_config": {
    "top_k": 8,
    "modality_weights": {
      "text": 0.7,
      "image": 0.3
    }
  }
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "answer": {
      "text": "基于多模态信息的详细回答...",
      "supporting_media": [
        {
          "type": "image",
          "content": "base64编码的相关图片或URL",
          "description": "图片描述",
          "relevance_score": 0.88
        }
      ]
    },
    "sources": [
      {
        "chunk_id": "chunk_001",
        "document_id": "doc_123456", 
        "modality": "text",
        "content": "引用的源文本内容...",
        "score": 0.92,
        "location": {
          "page": 5,
          "section": "方法论"
        }
      },
      {
        "chunk_id": "chunk_002",
        "document_id": "doc_123456",
        "modality": "image", 
        "content": "base64编码的源图片或URL",
        "caption": "实验数据图表",
        "score": 0.85
      }
    ],
    "metadata": {
      "session_id": "session_123456",
      "response_id": "resp_789012",
      "generation_time": "2.3s",
      "token_count": 456,
      "modalities_used": ["text", "image"]
    },
    "reasoning_chain": [
      {
        "step": 1,
        "action": "检索相关文本片段",
        "result": "找到5个相关文本块"
      },
      {
        "step": 2, 
        "action": "检索相关图片",
        "result": "找到3个相关图片"
      },
      {
        "step": 3,
        "action": "综合分析多模态信息",
        "result": "生成最终回答"
      }
    ]
  }
}
```

### 3.2 流式问答接口
```http
POST /api/v1/chat/stream
Content-Type: application/json
```

使用Server-Sent Events (SSE)进行流式响应，参数与多轮对话相同。

## 4. 错误处理

所有接口都遵循统一的错误响应格式：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "具体的错误描述",
    "details": {
      "field": "具体出错的字段",
      "reason": "具体原因"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 主要错误码：
- `INVALID_INPUT`: 输入参数错误
- `UNAUTHORIZED`: 未授权访问
- `RATE_LIMITED`: 请求频率超限
- `PROCESSING_ERROR`: 处理过程中出错
- `MODEL_UNAVAILABLE`: 模型服务不可用
- `STORAGE_ERROR`: 存储服务错误

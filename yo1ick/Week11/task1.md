# task1

## 多模态RAG项目接口设计

### 数据管理接口

| 接口名称 | 请求方法 | 路径 | 描述 |
| ------------ | ------------ | ------ | ------ |
| 文档上传接口 | POST | /api/v1/documents | 上传多模态文档并进行处理 |
| 文档列表接口 | GET | /api/v1/documents | 获取文档列表及基本信息 |
| 文档删除接口 | DELETE | /api/v1/documents/{doc_id} | 删除指定文档 |
| 文档更新接口 | PUT | /api/v1/documents/{doc_id} | 更新文档元数据 |

#### 文档上传接口

**请求参数**

| 参数名 | 类型 | 是否必填 | 描述 |
| --------- | ------ | ------------ | ------ |
| file | file | 是 | 多模态文件（支持pdf、docx、jpg、png等格式） |
| doc_name | string | 是 | 文档名称 |
| doc_type | string | 是 | 文档类型（text/image/table/mixed） |
| tags | array | 否 | 文档标签列表 |
| namespace | string | 否 | 命名空间，用于数据隔离 |

**返回结果**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "doc_id": "doc_123456",
    "doc_name": "产品说明书",
    "doc_type": "mixed",
    "size": 204800,
    "pages": 10,
    "status": "processing",
    "created_at": "2025-11-13T10:30:00Z"
  }
}
```

#### 文档列表接口

**请求参数**

| 参数名 | 类型 | 是否必填 | 描述 |
| --------- | ------ | ------------ | ------ |
| page | int | 否 | 页码，默认1 |
| page_size | int | 否 | 每页数量，默认20 |
| namespace | string | 否 | 按命名空间筛选 |
| tags | array | 否 | 按标签筛选 |
| start_time | string | 否 | 创建时间起始点（ISO格式） |
| end_time | string | 否 | 创建时间结束点（ISO格式） |

**返回结果**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 100,
    "page": 1,
    "page_size": 20,
    "documents": [
      {
        "doc_id": "doc_123456",
        "doc_name": "产品说明书",
        "doc_type": "mixed",
        "size": 204800,
        "pages": 10,
        "status": "completed",
        "tags": ["product", "manual"],
        "created_at": "2025-11-13T10:30:00Z"
      }
    ]
  }
}
```

### 多模态检索接口

| 接口名称 | 请求方法 | 路径 | 描述 |
| ------------ | ------------ | ------ | ------ |
| 文本检索接口 | POST | /api/v1/retrieve/text | 基于文本进行检索 |
| 图像检索接口 | POST | /api/v1/retrieve/image | 基于图像进行检索 |
| 混合检索接口 | POST | /api/v1/retrieve/mixed | 基于文本+图像进行混合检索 |

#### 混合检索接口

**请求参数**

| 参数名 | 类型 | 是否必填 | 描述 |
| --------- | ------ | ------------ | ------ |
| query | string | 是 | 检索文本 |
| image_url | string | 否 | 图像URL（base64编码或网络地址） |
| top_k | int | 否 | 返回结果数量，默认10 |
| namespace | string | 否 | 检索命名空间 |
| filter | object | 否 | 元数据过滤条件 |
| modalities | array | 否 | 指定检索模态类型，默认全部 |

**返回结果**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "query_id": "query_789012",
    "results": [
      {
        "doc_id": "doc_123456",
        "chunk_id": "chunk_345678",
        "modality": "text",
        "content": "产品尺寸为20x30x40cm",
        "score": 0.92,
        "page_num": 5,
        "position": {
          "x": 100,
          "y": 200,
          "width": 300,
          "height": 150
        },
        "metadata": {
          "timestamp": "2025-11-13T10:30:00Z",
          "tags": ["尺寸", "规格"]
        }
      },
      {
        "doc_id": "doc_123456",
        "chunk_id": "chunk_345679",
        "modality": "image",
        "content": "https://s.coze.cn/product_image_01.jpg",
        "score": 0.88,
        "page_num": 6,
        "position": {
          "x": 50,
          "y": 300,
          "width": 400,
          "height": 300
        },
        "metadata": {
          "description": "产品外观图",
          "tags": ["外观", "图片"]
        }
      }
    ]
  }
}
```

### 多模态问答接口

| 接口名称 | 请求方法 | 路径 | 描述 |
| ------------ | ------------ | ------ | ------ |
| 问答接口 | POST | /api/v1/qa | 基于多模态检索结果进行问答 |
| 对话历史接口 | GET | /api/v1/qa/history/{session_id} | 获取对话历史 |
| 对话结束接口 | POST | /api/v1/qa/end/{session_id} | 结束当前对话 |

#### 问答接口

**请求参数**

| 参数名 | 类型 | 是否必填 | 描述 |
| --------- | ------ | ------------ | ------ |
| query | string | 是 | 用户问题文本 |
| image_url | string | 否 | 问题相关图像 |
| session_id | string | 否 | 对话会话ID，不提供则创建新会话 |
| top_k | int | 否 | 检索结果数量，默认5 |
| response_type | string | 否 | 回答类型（text/image/mixed），默认text |
| stream | boolean | 否 | 是否流式返回，默认false |

**返回结果**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "session_id": "session_456789",
    "query_id": "query_789012",
    "answer": "产品尺寸为20x30x40cm，重量约2.5kg。以下是产品外观图：https://s.coze.cn/product_image_01.jpg",
    "reference": [
      {
        "doc_id": "doc_123456",
        "chunk_id": "chunk_345678",
        "page_num": 5,
        "content": "产品尺寸为20x30x40cm"
      }
    ],
    "answer_type": "mixed",
    "timestamp": "2025-11-13T14:30:00Z"
  }
}
```

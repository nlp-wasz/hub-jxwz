## 对多模态RAG的项目，milvus如果要存储文本、图、表，数据结构如何设计？

1. **统一管理**：不同类型数据在同一个集合中管理
2. **类型区分**：通过字段明确标识数据类型
3. **关联存储**：保持文档内不同元素的关联关系

## 数据结构设计方案

| 字段名 | 类型 | 说明 | 索引 |
|--------|------|------|------|
| **id** | String | 唯一标识符 | 主键 |
| **knowledge_base_id** | String | 所属知识库ID | 标量索引 |
| **document_id** | String | 所属文档ID | 标量索引 |
| **page_number** | Int | 页码 | 标量索引 |
| **chunk_type** | String | 数据类型：text/image/table | 标量索引 |

### 内容相关字段（根据chunk_type变化）

#### 文本类型 (chunk_type = "text")
| 字段名 | 类型 | 说明 |
|--------|------|------|
| **text_content** | String | 文本内容 |
| **bge_vector** | FloatVector | BGE编码的文本向量（768维） |
| **clip_text_vector** | FloatVector | CLIP文本编码向量（512维） |
| **metadata** | JSON | 文本元数据（字体、位置等） |

#### 图像类型 (chunk_type = "image")
| 字段名 | 类型 | 说明 |
|--------|------|------|
| **image_path** | String | 图像存储路径 |
| **image_caption** | String | 图像描述文本 |
| **clip_image_vector** | FloatVector | CLIP图像编码向量（512维） |
| **ocr_text** | String | OCR提取的文本 |
| **metadata** | JSON | 图像元数据（尺寸、位置等） |

#### 表格类型 (chunk_type = "table")
| 字段名 | 类型 | 说明 |
|--------|------|------|
| **table_structure** | JSON | 表格结构数据 |
| **table_text** | String | 表格文本内容 |
| **table_image_path** | String | 表格截图路径（可选） |
| **bge_vector** | FloatVector | 表格文本向量 |
| **clip_text_vector** | FloatVector | 表格描述向量 |
| **clip_image_vector** | FloatVector | 表格图像向量（如果有截图） |
| **metadata** | JSON | 表格元数据 |

### 关联关系字段
| 字段名 | 类型 | 说明 |
|--------|------|------|
| **parent_chunk_id** | String | 父chunk ID（用于嵌套结构） |
| **related_chunk_ids** | List[String] | 相关chunk IDs |
| **document_order** | Int | 在文档中的顺序 |

## 多模态RAG的项目，milvus如果要存储文本、图、表，数据结构如何设计？

### 1. 文本向量集合（text_vector_collection）
- 核心用途：存储文本chunk的BGE向量，支撑文本检索。
| 字段名               | 字段类型         | 核心用途                                                                 |
|----------------------|------------------|--------------------------------------------------------------------------|
| vector_id            | String（主键）   | Milvus侧唯一标识，格式：kb_[知识库ID]_txt_[文本元数据ID]                  |
| text_meta_id         | String（索引）   | 关联关系库的“文本元数据表”主键，用于查询元数据（如chunk内容、页码）       |
| knowledge_base_id    | String（索引）   | 快速过滤目标知识库，减少检索范围                                         |
| text_embedding       | FloatVector(768) | BGE生成的文本向量，核心检索字段                                          |

- 索引设计：`knowledge_base_id`建布尔索引（快速过滤），`text_embedding`建HNSW索引（高维向量检索）。

### 2. 图像向量集合（image_vector_collection）
- 核心用途：存储图像的CLIP向量，支撑图像检索/文本→图像检索。
| 字段名               | 字段类型         | 核心用途                                                                 |
|----------------------|------------------|--------------------------------------------------------------------------|
| vector_id            | String（主键）   | Milvus侧唯一标识，格式：kb_[知识库ID]_img_[图像元数据ID]                  |
| image_meta_id        | String（索引）   | 关联关系库的“图像元数据表”主键                                           |
| knowledge_base_id    | String（索引）   | 知识库过滤                                                               |
| image_embedding      | FloatVector(512) | CLIP生成的图像向量，核心检索字段                                          |

- 索引设计：同文本向量集合，保证检索效率一致。

### 3. 图表向量集合（chart_vector_collection）
- 核心用途：存储图表的CLIP向量，支撑图表检索/文本→图表检索。
| 字段名               | 字段类型         | 核心用途                                                                 |
|----------------------|------------------|--------------------------------------------------------------------------|
| vector_id            | String（主键）   | Milvus侧唯一标识，格式：kb_[知识库ID]_cht_[图表元数据ID]                  |
| chart_meta_id        | String（索引）   | 关联关系库的“图表元数据表”主键                                           |
| knowledge_base_id    | String（索引）   | 知识库过滤                                                               |
| chart_embedding      | FloatVector(512) | CLIP生成的图表视觉向量，核心检索字段                                      |

- 索引设计：新增`chart_meta_id`布尔索引，方便通过元数据ID反向查询向量。

# 【作业3】对多模态RAG的项目，milvus如果要存储文本、图、表，数据结构如何设计？

向量索引表
字段名	数据类型	说明
id	VARCHAR(200)	主键
embedding	FLOAT_VECTOR(1024)	统一检索向量
content_type	VARCHAR(50)	内容类型
content_id	VARCHAR(200)	外键，指向内容详情表
text_description	VARCHAR(4000)	检索用文本描述
source	VARCHAR(500)	数据来源


内容详情表
字段名	数据类型	说明
content_id	VARCHAR(200)	主键，与vector_index.content_id关联
content_type	VARCHAR(50)	内容类型
text_content	VARCHAR(20000)	文本内容全文
image_path	VARCHAR(500)	图像文件路径
image_base64	VARCHAR(100000)	小图像Base64编码（可选）
table_data	JSON	表格结构化数据
table_html	VARCHAR(10000)	表格HTML表示
embedding_model	VARCHAR(100)	使用的嵌入模型版本
processing_info	JSON	处理流水线信息
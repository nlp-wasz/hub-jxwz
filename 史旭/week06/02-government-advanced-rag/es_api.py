from elasticsearch import Elasticsearch

# 政企项目  Elasticsearch 文档数据库管理（连接es文档数据库，创建索引操作）  也可当做向量数据库使用

# 1.连接 es 向量数据库
es_url = "http://localhost:9200"
es = Elasticsearch(es_url)
print("***  es加载完成  ***")

# 2.创建 文档信息索引 和 文档chunk信息索引
# 文档信息索引
document_info_index = "document_info"
document_info_mappping = {
    "mappings": {
        "properties": {
            "document_id": {
                "type": "integer"
            },
            "document_title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "document_category": {
                "type": "keyword"
            },
            "knowledge_id": {
                "type": "integer"
            },
            "file_path": {
                "type": "text"
            },
            "abstract": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            }
        }
    }
}

# if es.indices.exists(index=document_info_index):
#     print(f"{document_info_index}索引已经存在")
# else:
#     es.indices.create(index=document_info_index, body=document_info_mappping)

# 文档chunk信息索引
document_chunk_info_index = "document_chunk_info"
document_chunk_info_mappping = {
    "mappings": {
        "properties": {
            "chunk_id": {
                "type": "integer"
            },
            "document_id": {
                "type": "integer"
            },
            "knowledge_id": {
                "type": "integer"
            },
            "page_number": {
                "type": "integer"
            },
            "chunk_content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,  # 建立索引提高检索效率
                "similarity": "cosine"  # 相似性检索方法（余弦相似度）
            }
        }
    }
}
# if es.indices.exists(index=document_chunk_info_index):
#     print(f"{document_chunk_info_index}索引已经存在")
# else:
#     es.indices.create(index=document_chunk_info_index, body=document_chunk_info_mappping)

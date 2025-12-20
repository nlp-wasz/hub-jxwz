import json

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 通过es 构建索引，存储向量信息

# sbert 编码模型
sbert_model = SentenceTransformer("../models/BAAI/bge-small-zh-v1.5")

# 1.创建 es 连接对象
es_url = 'http://localhost:9200'
es = Elasticsearch([es_url])

# 2.创建 指定索引
if es.ping():
    print("连接成功")
else:
    print("连接失败")

# 2.创建索引
index_name = "es_test_demo1"
mapping = {
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "text_vector": {
                "type": "dense_vector",  # 向量类型
                "dims": 512,  # 维度
                "index": True,  # 是否为向量建立索引，提交检索效率
                "similarity": "cosine"  # 检索向量相似度的方法（cosine余弦相似度  dot_product点积  l2_norm欧式距离）
            }
        }
    }
}
if es.indices.exists(index=index_name):
    print(f"索引{index_name}已经存在")
    es.indices.delete(index=index_name)
    print(f"索引{index_name}已经删除")
    res = es.indices.create(index=index_name, body=mapping)
    print(res)
    print(f"索引{index_name}创建成功")
else:
    res = es.indices.create(index=index_name, body=mapping)
    print(res)
    print(f"索引{index_name}创建成功")

# 3.添加数据
documents = [
    "人工智能是未来的趋势。",
    "机器学习是人工智能的一个重要分支。",
    "自然语言处理技术让机器理解人类语言。",
    "今天天气真好，适合出去玩。",
    "我最喜欢的运动是篮球和足球。"
]
for text in documents:
    # 对 文本 进行编码
    encode_text = sbert_model.encode(text)
    # 将原文本 和 编码后的文本 存入索引
    data = {"text": text, "text_vector": encode_text}
    es.index(index=index_name, document=data)

# 刷新
es.indices.refresh(index=index_name)

# 4.检索
print("\n--- 执行向量检索 ---")
query_text = "关于AI和未来的技术"

# 对检索文本 进行编码
encode_query_text = sbert_model.encode(query_text)

# ①向量检索
query1 = {
    "knn": {  # knn 检索方式
        "field": "text_vector",  # 制定检索字段
        "query_vector": encode_query_text,  # 指定检索向量
        "k": 3,  # 取 topK 文档信息
        "num_candidates": 10  # 候选topN，从候选10个文档中，选择top3
    },
    "_source": False,  # 不返回原数据信息
    "fields": ["text"]  # 指定返回 某些字段 信息（不在_source下显示，而是在fields下显示）
}
query1_res = es.search(index=index_name, body=query1)
print(f"查询文本: '{query_text}'")
print(json.dumps(query1_res["hits"]["hits"], indent=2, ensure_ascii=False))

print("-" * 100)

# ②向量检索 + query查询
query2 = {
    "knn": {
        "field": "text_vector",
        "query_vector": encode_query_text,
        "k": 3,
        "num_candidates": 10
    },
    "query": {
        "bool": {
            "must": {
                "match_all": {}
            },
            "filter": {
                "term": {
                    "text": "人工智能"
                }
            }
        }
    },
    "_source": False,  # 不返回原数据信息
    "fields": ["text"]  # 指定返回 某些字段 信息（不在_source下显示，而是在fields下显示）
}
query2_res = es.search(index=index_name, body=query2)
print(f"查询文本: '{query_text}'")
print(json.dumps(query2_res["hits"]["hits"], indent=2, ensure_ascii=False))

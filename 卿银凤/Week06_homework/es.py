from datetime import datetime
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import time

# 连接到 Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# 检查连接是否成功
if es.ping():
    print("成功连接到 Elasticsearch")
else:
    print("无法连接到 Elasticsearch，请确保服务正在运行")
    exit(1)

# 索引名称
INDEX_NAME = "data_experiment"

# 删除已存在的索引（用于实验）
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

# 创建索引映射
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "views": {"type": "integer"},
            "timestamp": {"type": "date"},
            "tags": {"type": "keyword"},
            "is_published": {"type": "boolean"}
        }
    }
}

# 创建索引
es.indices.create(index=INDEX_NAME, body=mapping)
print(f"已创建索引: {INDEX_NAME}")

# 准备不同类型的数据
documents = [
    {
        "title": "Python 编程入门",
        "content": "Python 是一种高级编程语言，以其清晰的语法和代码可读性而闻名。",
        "views": 150,
        "timestamp": datetime.now(),
        "tags": ["编程", "Python", "教程"],
        "is_published": True
    },
    {
        "title": "数据科学实战",
        "content": "数据科学结合了统计学、数据分析、机器学习等相关方法，旨在从数据中提取知识和见解。",
        "views": 275,
        "timestamp": datetime.now(),
        "tags": ["数据科学", "机器学习", "分析"],
        "is_published": True
    },
    {
        "title": "Elasticsearch 指南",
        "content": "Elasticsearch 是一个分布式的搜索和分析引擎，可以用于各种类型的搜索应用。",
        "views": 89,
        "timestamp": datetime.now(),
        "tags": ["Elasticsearch", "数据库", "搜索"],
        "is_published": False
    },
    {
        "title": "Web 开发技术",
        "content": "现代 Web 开发涉及多种技术和框架，包括前端和后端开发。",
        "views": 420,
        "timestamp": datetime.now(),
        "tags": ["Web开发", "前端", "后端"],
        "is_published": True
    }
]

# 批量插入数据
actions = [
    {
        "_index": INDEX_NAME,
        "_source": doc
    }
    for doc in documents
]

success, failed = bulk(es, actions)
print(f"成功插入 {success} 条文档，失败 {failed} 条")

# 等待索引刷新
es.indices.refresh(index=INDEX_NAME)
time.sleep(1)  # 确保数据已索引

print("\n=== 数据检索 ===\n")

# 1. 简单全文搜索
print("1. 搜索包含'Python'的文档:")
query = {
    "query": {
        "match": {
            "content": "Python"
        }
    }
}
result = es.search(index=INDEX_NAME, body=query)
for hit in result['hits']['hits']:
    print(f"标题: {hit['_source']['title']}, 得分: {hit['_score']:.2f}")

# 2. 范围查询
print("\n2. 搜索浏览量大于100的文档:")
query = {
    "query": {
        "range": {
            "views": {
                "gt": 100
            }
        }
    }
}
result = es.search(index=INDEX_NAME, body=query)
for hit in result['hits']['hits']:
    print(f"标题: {hit['_source']['title']}, 浏览量: {hit['_source']['views']}")


print("\n=== 完成 ===")
# SemanticRouter （快速实现意图识别）
# 使用 Redis 存储 标签和对应的实例文本
import hashlib
import traceback

import numpy as np
from redis import Redis
from typing import Optional, List, Union
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("../../../models/BAAI/bge-small-zh-v1.5")


class SemanticRouter:
    def __init__(self,
                 group_key: Optional[str] = None,
                 localhost: Optional[str] = "localhost",
                 port: Optional[int] = 6379,
                 prompt: Union[str, List[str]] = None,
                 ttl: Optional[int] = 3600):
        # 创建redis对象
        self.redis = Redis(host=localhost, port=port, db=1, decode_responses=True)
        self.group_key = group_key
        self.prompt = prompt
        self.ttl = ttl
        self.es = Elasticsearch("http://localhost:9200")
        self.index_name = "semantic_router"

    # 创建 es 索引
    def create_index(self):
        # 存储 标准答案和对应的编码信息 的索引文档
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        # 创建 索引
        mapping = {
            "mappings": {
                "properties": {
                    "query": {
                        "type": "keyword"
                    },
                    "query_embedding": {
                        "type": "dense_vector",  # 向量类型
                        "dims": 512,  # 维度
                        "index": True,  # 是否为向量建立索引，提高检索效率
                        "similarity": "cosine"  # 检索向量相似度的方法（cosine余弦相似度  dot_product点积  l2_norm欧式距离）
                    }
                }
            }
        }

        return self.es.indices.create(index=self.index_name, body=mapping)

    # es 添加数据
    def add_index(self, query: str, query_embedding):
        # 构建 es 文档数据
        es_data = {
            "query": query,
            "query_embedding": query_embedding
        }
        return self.es.index(index=self.index_name, body=es_data)

    # query_embedding 向量检索
    def search_query_embedding(self, query: str):
        # 查询 query字段值 为 query 的 信息
        query_embedding = model.encode(query)
        mapping = {
            "knn": {
                "field": "query_embedding",
                "query_vector": query_embedding,
                "k": 1,
                "num_candidates": 3
            },
            "_source": False,
            "fields": ["query"]
        }

        search_res = self.es.search(index=self.index_name, body=mapping)

        return search_res['hits']['hits']

    # 对用户问题进行检索，判断是否存在一样的问题
    def user_query_search(self, query: str):
        # 判断是否进行 向量匹配
        query_embedding_res = self.search_query_embedding(query)
        # 根据 es 检索结果，从 redis 缓存中获取结果
        key = query_embedding_res[0]["fields"]["query"][0]
        res = self.redis.get(f"{self.group_key}:{key}")

        return res

    # 往 Redis 和 Es 中添加信息（标准问题和标准答案）
    def add_redis_es(self, messages: List[dict]):
        # 循环添加问题
        es_data = []
        redis_data = []
        for message in messages:
            # 获取 问题 和 答案
            query = message["query"]
            target = message["target"]

            # 对 query 进行编码
            query_embedding = model.encode(query)

            es_data.append({"query": query, "query_embedding": query_embedding})
            redis_data.append({"key": f"{self.group_key}:{query}", "target": target})

        # 存储信息
        try:
            for e, r in zip(es_data, redis_data):
                # es（通过 id 去重）
                document_id = hashlib.md5(e.get("query").strip().encode("utf-8")).hexdigest()
                self.es.index(index=self.index_name, id=document_id, body=e)

                # redis
                self.redis.setex(r.get("key"), self.ttl, r.get("target"))

            return "添加成功！"
        except Exception as e:
            traceback.print_exc()
            return "添加失败！"


sc = SemanticRouter(
    group_key="SemanticRouter",
)

# print(sc.create_index())

messages = [
    {"query": "hello", "target": "greeting"},
    {"query": "hi", "target": "greeting"},
    {"query": "bye", "target": "farewell"},
    {"query": "goodbye", "target": "farewell"},
    {"query": "goodbye", "target": "farewell"}  # 重复（es去重，根据id）
]
# print(sc.add_redis_es(messages))
print(sc.user_query_search("hello world"))
print(sc.user_query_search("你好"))
print(sc.user_query_search("再见"))

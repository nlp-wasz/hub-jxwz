import json

from elasticsearch import Elasticsearch

# 1.创建 es 连接对象
es_url = "http://localhost:9200"
es = Elasticsearch(es_url)

if es.ping():
    print("连接成功")
else:
    print("连接失败")

# 2.创建索引
index_name = "es_test_demo1"
mapping = {
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "ik_max_word"},
            "description": {"type": "text", "analyzer": "ik_smart"},
            "price": {"type": "float"},
            "category": {"type": "keyword"},
            "stock": {"type": "integer"},
            "on_sale": {"type": "boolean"},
            "created_at": {"type": "date"}
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

# 3.添加文档信息
doc_1 = {
    "product_id": "A001",
    "name": "智能手机",
    "description": "最新款智能手机，性能强大，拍照清晰。",
    "price": 4999.50,
    "category": "电子产品",
    "stock": 150,
    "on_sale": True,
    "created_at": "2023-01-15T09:00:00Z"
}
es.index(index=index_name, id="A001", document=doc_1)
print("文档 'A001' 已插入。")

doc_2 = {
    "product_id": "B002",
    "name": "无线蓝牙耳机",
    "description": "音质卓越，佩戴舒适，超长续航。",
    "price": 699.00,
    "category": "电子产品",
    "stock": 300,
    "on_sale": True,
    "created_at": "2023-02-20T14:30:00Z"
}
es.index(index=index_name, id="B002", document=doc_2)
print("文档 'B002' 已插入。")

# refresh 刷新
es.indices.refresh(index=index_name)

# 4.查询
# ①全文检索“智能”
print("\n--- 检索 1: 全文检索“智能” ---")
query1 = {
    "query": {
        # "match": { # match 仅支持 一个字段的条件查询
        #     "name": "智能",
        #     "description": "智能"
        # },
        "multi_match": {  # 多字段 条件查询
            "query": "智能",  # 查询条件
            "fields": ["name", "description"]  # 匹配的字段
        }
    }
}
query1_res = es.search(index=index_name, body=query1)
print(query1_res)
for hit in query1_res["hits"]["hits"]:
    print(f"source：{hit['_source']} -- product_id：{hit['_source']['product_id']} -- name：{hit["_source"]["name"]} -- ")
print(json.dumps(query1_res["hits"]["hits"], indent=2, ensure_ascii=False))

# ② 全文检索 + 精确匹配
print("\n--- 检索 2: 结合查询与过滤  搜索价格低于 1000 元且正在促销的电子产品 ---")
query2 = {
    "query": {
        "bool": {
            "must": {
                "match_all": {  # match_all 检索全部

                }
            },
            "filter": [
                {"term": {"on_sale": True}},  # 精确匹配 条件1（必须 促销）
                {"term": {"category": "电子产品"}},  # 精确匹配 条件2（必须 电子产品）
                {"range": {"price": {"lt": 1000}}}  # 精确匹配 条件3（必须 小于1000元）
            ]
        }
    }
}
query2_res = es.search(index=index_name, body=query2)
print(query2_res)
for hit in query2_res["hits"]["hits"]:
    print(f"source：{hit['_source']} -- product_id：{hit['_source']['product_id']} -- name：{hit["_source"]["name"]} -- ")
print(json.dumps(query1_res["hits"]["hits"], indent=2, ensure_ascii=False))

# ③按关键词分组聚合
# 统计不同类别的商品数量
print("\n--- 检索 3: 聚合查询（按类别统计） ---")
query3 = {
    "aggregations": {
        "category-group": {
            "terms": {
                "field": "category",
                "size": 10
            }
        }
    },
    # "size":0
}
query3_res = es.search(index=index_name, body=query3)
# print(query3_res)
for agg in query3_res["aggregations"]["category-group"]["buckets"]:
    # 获取 聚合分组 结果
    print(f"category:{agg["key"]} -- doc_count:{agg["doc_count"]}")

print(json.dumps(query3_res["aggregations"]["category-group"]["buckets"], indent=2, ensure_ascii=False))

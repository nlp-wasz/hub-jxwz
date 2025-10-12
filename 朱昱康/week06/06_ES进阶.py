import json
import random
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch

def main():
    # 连接配置
    es = Elasticsearch(
        ["http://localhost:9200"],
        http_auth=("elastic", "Tdr2jRHS")
    )
    
    if not es.ping():
        print("连接失败")
        return
    
    print("连接成功")
    
    index_name = "products_simple"
    
    # 创建索引
    if not es.indices.exists(index=index_name):
        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    "price": {"type": "float"},
                    "category": {"type": "keyword"},
                    "brand": {"type": "keyword"},
                    "stock": {"type": "integer"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print("索引已创建")
    
    # 插入测试数据
    products = [
        {"name": "iPhone 15", "price": 5999, "category": "手机", "brand": "Apple", "stock": 100},
        {"name": "MacBook Pro", "price": 12999, "category": "电脑", "brand": "Apple", "stock": 50},
        {"name": "小米14", "price": 3999, "category": "手机", "brand": "Xiaomi", "stock": 200},
        {"name": "华为MateBook", "price": 6999, "category": "电脑", "brand": "Huawei", "stock": 80}
    ]
    
    for i, product in enumerate(products):
        es.index(index=index_name, id=f"prod{i+1}", body=product)
    
    es.indices.refresh(index=index_name)
    print("数据已插入")
    
    # 查询演示
    
    # 1. 搜索手机
    res = es.search(index=index_name, body={
        "query": {"match": {"category": "手机"}},
        "size": 10
    })
    print(f"手机商品: {len(res['hits']['hits'])}个")
    
    # 2. 价格范围查询
    res = es.search(index=index_name, body={
        "query": {
            "range": {"price": {"gte": 3000, "lte": 8000}}
        },
        "size": 10
    })
    print(f"价格3000-8000: {len(res['hits']['hits'])}个")
    
    # 3. 按品牌统计
    res = es.search(index=index_name, body={
        "size": 0,
        "aggs": {"brands": {"terms": {"field": "brand", "size": 10}}}
    })
    
    print("品牌统计:")
    for bucket in res['aggregations']['brands']['buckets']:
        print(f"  {bucket['key']}: {bucket['doc_count']}")

if __name__ == "__main__":
    main()

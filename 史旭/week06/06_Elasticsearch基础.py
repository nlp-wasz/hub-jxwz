import datetime

from elasticsearch import Elasticsearch

# Elasticsearch 使用基础（创建索引，添加文档信息，查询文档信息等  crud）

# 1.通过 elasticsearch 库，连接es数据库
es_url = "http://localhost:9200"
es_client = Elasticsearch(es_url)

if es_client.ping:
    print("连接成功")
else:
    print("连接失败")

# 2.创建 索引（索引名称 和 索引结构）
# ①索引结构为 json 类型（字段的分词器类型，在创建索引时设置，不可再修改，除非删除索引，重新创建）
index_name = "es_test_demo1"
mapping = {
    "mappings": {  # 必须，索引的结构信息
        "properties": {  # 索引的属性信息，即字段
            "title": {  # title 字段
                "type": "text",  # 文本类型
                "analyzer": "ik_max_word",  # 索引分词器（最细化分词器）
                "search_analyzer": "ik_smart"  # 搜索分词器（粗化分词器）
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "tags": {
                "type": "keyword"  # keyword 关键字类型（用于精确检索）
            },
            "author": {
                "type": "keyword"
            },
            "created_dt": {
                "type": "date"
            }
        }
    }
}

# ②首先判断 索引是否已经存在（不存在创创建）
if es_client.indices.exists(index=index_name):
    print(f"索引{index_name}已经存在")
    es_client.indices.delete(index=index_name)
    print(f"索引{index_name}已经删除")
    res = es_client.indices.create(index=index_name, body=mapping)
    print(f"索引{index_name}创建成功")
else:
    res = es_client.indices.create(index=index_name, body=mapping)
    print(f"索引{index_name}创建成功")

print("=" * 100)

# 3.添加 文档信息 到指定索引（需要 josn 类型信息）
# ①构建 文档信息
doc = [
    {
        "title": "Elasticsearch 入门指南",
        "content": "这是一篇关于如何安装和使用 Elasticsearch 的详细文章。学习搜索技术可以提升数据处理能力。",
        "tags": ["Elasticsearch", "教程", "搜索"],
        "author": ["张三", "李四"],
        "created_dt": datetime.datetime(2025, 9, 16, 11, 10, 50)
    },
    {
        "title": "深入理解IK分词器",
        "content": "IK分词器是中文分词的优秀工具。它的智能分词和最细粒度分词模式各有优势。",
        "tags": ["分词", "IK", "中文"],
        "author": "王五",
        "created_dt": datetime.datetime(2025, 9, 16, 11, 10, 50)
    }
]

# ②通过 index() 方法查询指定索引，并可指定doc参数，添加文档信息
for i, document in enumerate(doc):
    # index 索引名称， id（可选） 指定文档信息ID  document 要存储的文档信息
    es_client.index(index=index_name, id=i, document=document)

    print(f"文档 【{document.get('title')}】 已添加至索引{index_name}")

# ③（添加后如果不 refresh，可能会导致不能即使获取，es自动1秒刷新一次）
es_client.indices.refresh(index=index_name)

# 4.查询添加的信息
print("\n---  查询标题中的 '入门指南' ---")
# ①构建 查询 语句（json格式）
query1 = {
    "query": {  # query 代表查询
        "match": {  # match 代表要满足的的条件
            "title": "入门指南"  # 查询条件为 title字段必须包含 ‘入门指南’ 四个字
        }
    }
}

# ②search() 指定索引 根据条件 查询数据
query1_res = es_client.search(index=index_name, body=query1)
# print(query1_res)
for hit in query1_res["hits"]["hits"]:
    print(f"_score:{hit["_score"]}")
    print(f"title:{hit["_source"]["title"]}")
    print(f"content:{hit["_source"]["content"]}")
    print(f"tags:{hit["_source"]["tags"]}")
    print(f"author:{hit["_source"]["author"]}")
    print(f"created_dt:{hit["_source"]["created_dt"]}")
    print("-" * 50)

# 5.全文搜索 + 精确匹配
print("\n---  结合全文（搜索技术）和精确匹配（作者：张三） ---")
query2 = {
    "query": {
        "bool": {
            "must": {  # 必须满足的条件
                "match": {  # 全文检索
                    "content": "搜索技术"
                }
            },
            "filter": {  # 过滤条件
                "term": {  # 精确检索
                    "author": "张三"
                }
            }
        }
    }
}

query2_res = es_client.search(index=index_name, body=query2)
# print(query2_res)
for hit in query2_res["hits"]["hits"]:
    print(f"_score:{hit["_score"]}")
    print(f"title:{hit["_source"]["title"]}")
    print(f"content:{hit["_source"]["content"]}")
    print(f"tags:{hit["_source"]["tags"]}")
    print(f"author:{hit["_source"]["author"]}")
    print(f"created_dt:{hit["_source"]["created_dt"]}")
    print("-" * 50)

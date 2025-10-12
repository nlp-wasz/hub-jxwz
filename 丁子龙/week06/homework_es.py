# pip install elasticsearch
from elasticsearch import Elasticsearch

# 替换为你的 Elasticsearch 地址
ELASTICSEARCH_URL = "http://localhost:9200"

# 如果没有安全认证，直接创建客户端
es_client = Elasticsearch(ELASTICSEARCH_URL)

# 测试连接
if es_client.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")

# 定义索引名称和映射
index_name = "blog_posts_py"
mapping = {
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart"
      },
      "content": {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart"
      },
      "tags": { "type": "keyword" },
      "author": { "type": "keyword" },
      "created_at": { "type": "date" }
    }
  }
}

# 检查索引是否存在，存在则删除重建
if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name)
    print(f"旧索引 '{index_name}' 已删除。")

# 检查索引是否存在，如果不存在则创建
if not es_client.indices.exists(index=index_name):
    es_client.indices.create(index=index_name, body=mapping)
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已经存在。")

from datetime import datetime

# 模拟 10 篇博客文章
documents = [
    {
        "title": "Elasticsearch 入门指南",
        "content": "这是一篇关于如何安装和使用 Elasticsearch 的详细文章。学习搜索技术可以提升数据处理能力，适合初学者。",
        "tags": ["Elasticsearch", "教程", "搜索"],
        "author": "张三",
        "created_at": datetime(2023, 10, 26, 10, 0, 0)
    },
    {
        "title": "深入理解IK分词器",
        "content": "IK分词器是中文分词的优秀工具。它的智能分词和最细粒度分词模式各有优势，广泛应用于搜索引擎中。",
        "tags": ["分词", "IK", "中文", "NLP"],
        "author": "李四",
        "created_at": datetime(2023, 10, 25, 15, 30, 0)
    },
    {
        "title": "使用Kibana可视化日志数据",
        "content": "Kibana 是 Elastic Stack 的重要组成部分，可以帮助开发者通过图表和仪表板直观地分析日志与指标。",
        "tags": ["Kibana", "可视化", "日志分析"],
        "author": "王五",
        "created_at": datetime(2023, 11, 5, 9, 15, 0)
    },
    {
        "title": "Elasticsearch 性能调优实战",
        "content": "在大数据量场景下，合理设置分片数量、刷新间隔和查询方式能显著提升 Elasticsearch 查询性能。",
        "tags": ["性能优化", "大数据", "Elasticsearch"],
        "author": "张三",
        "created_at": datetime(2024, 1, 12, 14, 20, 0)
    },
    {
        "title": "Logstash 数据采集入门",
        "content": "Logstash 能够从多种来源收集日志数据，并进行过滤转换后发送到 Elasticsearch，是构建日志系统的利器。",
        "tags": ["Logstash", "数据采集", "ETL"],
        "author": "赵六",
        "created_at": datetime(2022, 12, 8, 11, 45, 0)
    },
    {
        "title": "全文搜索引擎对比：ES vs Solr",
        "content": "Elasticsearch 和 Apache Solr 都基于 Lucene 构建，但在易用性、分布式支持方面存在差异，适用于不同场景。",
        "tags": ["搜索引擎", "对比", "Lucene"],
        "author": "陈老师",
        "created_at": datetime(2024, 2, 20, 16, 10, 0)
    },
    {
        "title": "如何在 Python 中操作 Elasticsearch",
        "content": "通过 elasticsearch-py 客户端库，Python 开发者可以轻松实现索引管理、文档增删改查和复杂查询功能。",
        "tags": ["Python", "API", "开发"],
        "author": "张三",
        "created_at": datetime(2024, 3, 3, 8, 30, 0)
    },
    {
        "title": "Elastic Stack 在运维监控中的应用",
        "content": "结合 Beats、Logstash、Elasticsearch 和 Kibana，可构建完整的运维监控平台，实现实时告警与分析。",
        "tags": ["运维", "监控", "ELK"],
        "author": "王五",
        "created_at": datetime(2023, 9, 14, 13, 50, 0)
    },
    {
        "title": "倒排索引原理详解",
        "content": "倒排索引是搜索引擎的核心技术之一，它将文档中的词语映射到包含它们的文档列表，极大提升了检索效率。",
        "tags": ["原理", "倒排索引", "搜索算法"],
        "author": "陈老师",
        "created_at": datetime(2024, 4, 10, 10, 0, 0)
    },
    {
        "title": "利用 synonyms 实现同义词搜索",
        "content": "通过配置同义词词典，可以让用户搜索‘电脑’时也能匹配到‘计算机’相关内容，提升搜索体验。",
        "tags": ["同义词", "搜索优化", "用户体验"],
        "author": "李四",
        "created_at": datetime(2024, 5, 18, 17, 25, 0)
    }
]

for doc in documents:
    es_client.index(index=index_name, body=doc)
    print(f"文档已插入: '{doc['title']}'")

# 刷新索引，确保文档可被搜索到
es_client.indices.refresh(index=index_name)

# 定义查询函数
def search_docs(query):
    response = es_client.search(index=index_name, body=query)
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档：{hit['_source']['title']}")

# 1. 查询标题中的 "入门指南"
print("\n--- 1. 查询标题中的 '入门指南' ---")
query_1 = {
  "query": {
    "match": {
      "title": "入门指南"
    }
  }
}
search_docs(query_1)

# 2. 结合全文和精确匹配查询
print("\n--- 2. 结合全文（搜索技术）和精确匹配（作者：张三） ---")
query_2 = {
  "query": {
    "bool": {
      "must": {
        "match": {
          "content": "搜索技术"
        }
      },
      "filter": {
        "term": {
          "author": "张三"
        }
      }
    }
  }
}
search_docs(query_2)

print("\n--- 扩展测试：查找标签为 'Elasticsearch' 的文章 ---")
query_3 = {
    "query": {
        "term": {
            "tags": "Elasticsearch"
        }
    }
}
search_docs(query_3)

print("\n--- 扩展测试：2024 年发布的文章 ---")
query_4 = {
    "query": {
        "range": {
            "created_at": {
                "gte": "2024-01-01"
            }
        }
    }
}
search_docs(query_4)

print("\n--- 扩展测试：作者为 '李四' 且内容含 '分词' 的文章 ---")
query_5 = {
    "query": {
        "bool": {
            "must": { "match": { "content": "分词" } },
            "filter": { "term": { "author": "李四" } }
        }
    }
}
search_docs(query_5)
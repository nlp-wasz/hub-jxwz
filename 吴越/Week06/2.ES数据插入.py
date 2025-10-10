from elasticsearch import Elasticsearch

ELASTICSEARCH_URL="http://localhost:9200"

es_client=Elasticsearch(ELASTICSEARCH_URL)

if es_client.ping():
    print("连接成功！")
else:
    print("连接失败，检查lasticsearch服务是否运行！")


#定义索引名和映射
index_name="products"
mapping={
    "settings":{
        "number_of_shards":1,
        "number_of_replicas":0
    },

    "mappings":{
        "properties":{
            "product_id":{"type":"keyword"},
            "name":{"type":"text","analyzer":"ik_max_word","search_analyzer": "ik_smart"},
            "description":{"type":"text","analyzer":"ik_max_word","search_analyzer": "ik_smart"},
            "price":{"type":"float"},
            "category":{"type":"text","analyzer":"ik_max_word","search_analyzer": "ik_smart"},
            "stock":{"type":"integer"},
            "on_sale":{"type":"boolean"},
            "created_at":{"type":"date"}
        }
    }
}

if not es_client.indices.exists(index=index_name):
    es_client.indices.create(index=index_name, body=mapping)
    print(f"索引‘{index_name}’ 创建成功。")
else:
    print(f"索引‘{index_name}’ 已经存在。")

documents=[
    {
        "product_id":"A001",
        "name":"小米手机",
        "description":"最新款智能手机，性能强大，拍照清晰",
        "price":5000.0,
        "category":"电子产品",
        "stock":100,
        "on_sale":True,
        "created_at":"2018-08-01"
    },

    {
            "product_id":"A002",
            "name":"华为手机",
            "description":"华为自研麒麟芯片系列，高性能、低功耗",
            "price":69999.0,
            "category":"电子产品",
            "stock":200,
            "on_sale":True,
            "created_at":"2020-10-23"
    },
    {
                "product_id":"A003",
                "name":"苹果手机",
                "description":"高性能硬件、直观操作系统",
                "price":69999.0,
                "category":"电子产品",
                "stock":300,
                "on_sale":True,
                "created_at":"2020-10-23"
            },
]

for doc in documents:
    es_client.index(index=index_name,document=doc)
    print(f'文档已插入：{doc["product_id"]}')

es_client.indices.refresh(index=index_name)

response=es_client.search(index=index_name,body={
    "query":{
        "match":{
            "description":"麒麟芯片"
        }
    }
})

for hit in response['hits']['hits']:
  print(hit['_source']['name'])



import requests

# 测试 elasticsearch 是否能够正常连接

# 1.测试连接
print("--- 正在测试 Elasticsearch 连接 ---")

# ①本次连接地址（开启身份认证时 使用https）
# es_url = "https://localhost:9200"
es_url = "http://localhost:9200"

# ②身份认证
# auth = ("elastic", "shixu20020601")
auth = None

# ③关闭 SSL 验证
verif_ssl = False

# ④发送 get 请求，验证是否能够连接（设置请求头， 无请求体数据信息）
header = {"Content-Type": "application/json"}
res = requests.get(es_url, headers=header, auth=auth, verify=verif_ssl)
print(f"es本地连接--状态：{res.status_code}")
print(f"es本地连接--结果：{res.json()}")

print("=" * 100)

# 2.测试es内置分词器（英文分词器，对中文不友好）
print("\n--- 正在测试常见的 Elasticsearch 内置分词器 ---")

es_analyzers = ["standard", "simple", "whitespace", "english"]
test_text = "Hello, world! This is a test."
for analyzer in es_analyzers:
    print(f"\n使用分词器：{analyzer}")

    # 设置 请求体
    data = {"analyzer": analyzer, "text": test_text}
    # 发送请求（包含请求体，需要发送post请求，并且地址为es分词器 对应的url）
    es_analyzer_url = f"{es_url}/_analyze"
    es_analyzer_res = requests.post(es_analyzer_url, headers=header, json=data, auth=auth, verify=verif_ssl)

    print(f"es内置分词器连接--状态：{es_analyzer_res.status_code}")
    print(f"es内置分词器连接--结果：{es_analyzer_res.json()}")
    if es_analyzer_res and ("tokens" in es_analyzer_res.json()):
        analyze_text = [tokens["token"] for tokens in es_analyzer_res.json()["tokens"]]
        print(f"原始文本：{test_text}")
        print(f"分词后的文本：{analyze_text}")

print("=" * 100)

# 3.测试 安装的 ik分词器（对中文进行分词， 需要先下载安装包，然后解压到es的plugins文件夹）
print("\n--- 正在测试安装的 IK 中文分词器 ---")
ik_analyzers = ["ik_smart", "ik_max_word"]
test_text_zh = "我在使用Elasticsearch，这是我的测试。"

for analyzer in ik_analyzers:
    print(f"\n使用分词器：{analyzer}")

    # 构建请求体
    data = {"analyzer": analyzer, "text": test_text_zh}
    # 发送请求
    es_analyzer_url = f"{es_url}/_analyze"
    es_analyzer_res = requests.post(es_analyzer_url, headers=header, json=data, auth=auth, verify=verif_ssl)

    print(f"es内置分词器连接--状态：{es_analyzer_res.status_code}")
    print(f"es内置分词器连接--结果：{es_analyzer_res.json()}")

    if es_analyzer_res and "tokens" in es_analyzer_res.json():
        analyze_text = [tokens.get("token") for tokens in es_analyzer_res.json()["tokens"]]
        print(f"原始文本：{test_text_zh}")
        print(f"分词后的文本：{analyze_text}")

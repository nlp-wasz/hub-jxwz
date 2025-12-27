from openai import OpenAI

# 1.使用 OpenAi 调用 DeepSeek
deepseek = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="sk-c77746bd2b844d2784b361ba269bf110"
)

# 创建聊天对象，并得到返回输出
response = deepseek.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个非常专业且全面的 assistant"},
        {"role": "user", "content": "现在是2025年9月10号，今天的天气如何，以及产生的影响？"}
    ]
)

print(response.json)
print(response.choices[0].message)
print(response.choices[0].message.content)

print("-" * 100)

# HTTP 方式调用（DeepSeek未公开 HTTP接口）
# import requests
#
# deepseek_url = "https://api.deepseek.com"
#
# header = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer sk-c77746bd2b844d2784b361ba269bf110"
# }
#
# data = {
#     "model": "deepseek-chat",
#     "messages": [
#         {"role": "system", "content": "你是一个专业的气象学专家"},
#         {"role": "user", "content": "今天的天气如何，以及产生的影响？"}
#     ]
# }
#
# response = requests.post(deepseek_url, headers=header, json=data)
# print(response.json())

from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API 地址
    api_key="not_required"  # Ollama 默认无需真实 API Key，填任意值即可
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "周杰伦是谁？"}
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512  # 最大生成 token 数
)
# 打印结果
print(response.choices[0].message.content)

print("-" * 50)

# HTTP 方式
import requests

qwen_url = "http://localhost:11434/api/generate"

header = {
    "Content-Type": "application/json",
    "Authorization": "123"
}

data = {
    "model": "qwen3:0.6b",
    "prompt": "周杰伦是谁？",
    "stream": False
}

response = requests.post(url=qwen_url, headers=header, json=data)
print(response)
print(response.json()["response"])

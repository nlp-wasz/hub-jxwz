# 本地安装ollama工具，尝试在本地部署一个qwen3:0.6b模型，调用本地模型

import openai

base_url = "http://localhost:11434/v1"
api_key = "sk-qwertyuiop"

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model="qwen3:0.6b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"},
    ],
    temperature=0.9,
    max_tokens=1024,
    stream=True,
)

print(response.choices[0].message.content)

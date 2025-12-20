from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        # {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，什么是大模型？"}
    ],
    temperature=0.2,  # 控制生成多样性
    max_tokens=512  # 最大生成 token 数
)


# 打印结果
print(response.choices[0].message.content)

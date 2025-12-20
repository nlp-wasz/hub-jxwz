# 申请deepseek的api，https://platform.deepseek.com/usage，使用openai库调用大模型


import openai

# api_key = "sk-931f9a85c33b44878d3b7e5a22cfe537"
api_key = "sk-119253c0b5c647e291e54b7437b044bf"
api_base = "https://api.deepseek.com"

client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base,
)

res = client.chat.completions.create(
    model='deepseek-chat',
    stream=True,
    max_tokens=1024,
    temperature=0.9,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
)
for chunk in res:
    print(chunk.choices[0].delta.content, end="", flush=True)

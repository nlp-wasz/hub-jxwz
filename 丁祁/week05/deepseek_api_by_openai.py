
import openai

# 作业1: 申请deepseek的api，https://platform.deepseek.com/usage， 使用openai 库调用云端大模型。
client = openai.OpenAI(
    api_key="sk-ed8a2b3087c64b26bcdee7012dda7718",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)


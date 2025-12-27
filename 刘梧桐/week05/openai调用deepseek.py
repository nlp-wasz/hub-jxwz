from openai import OpenAI

client = OpenAI(
    api_key="sk-ff121d59ae604ee5b859ebb2db2fa8d0",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role":"system","content":"you are a helpful assistant"},
        {"role":"user","content":"introduce yourself"}
    ],
    temperature=0.4,
    max_tokens=512
)
print(response.choices[0].message.content)

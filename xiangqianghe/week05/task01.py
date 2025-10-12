from openai import OpenAI

client = OpenAI(api_key="sk-3cfedc9d947f42caa18325999ba67723", base_url="https://api.deepseek.com/v1")

stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁？请回答你使用的基座模型及使用到的模型技术"},
    ],
    stream=True
)

full_response = ""
for chunk in stream:
    # 检查是否有内容增量
    if chunk.choices[0].delta.content is not None:
        # 获取当前块的内容
        content_chunk = chunk.choices[0].delta.content
        # 打印当前块的内容（实时显示）
        print(content_chunk, end='', flush=True)
        # 添加到完整响应中
        full_response += content_chunk

# 打印完整的响应
print("\n\n完整响应:")
print(full_response)


# 当你设置 stream=True时，API 返回的是一个生成器对象（Stream），而不是一个完整的响应对象。因此你不能直接访问 response.choices。
# print(response.choices[0].message.content)

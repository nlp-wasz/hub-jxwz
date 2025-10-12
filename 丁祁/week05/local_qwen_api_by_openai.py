
import openai

# 作业2: 本地安装ollama工具，尝试在本地部署一个qwen3:0.6b模型
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=""
)


response = client.chat.completions.create(
    model="qwen3:0.6b",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，国庆节北京周边有哪些地方值得去？ 给出明细的说明"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)



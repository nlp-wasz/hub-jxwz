import openai
import json

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
# deepseek
client = openai.OpenAI(
    api_key="sk-dd021faa650d4c84a5b8714aaab3b2d4", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://api.deepseek.com",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
)
print(completion.model_dump_json())

# https://platform.deepseek.com/usage

# https://platform.moonshot.cn/docs/introduction

# http://bigmodel.cn/
import requests
import json

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-3cfedc9d947f42caa18325999ba67723"
}
data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "作业3: 现在有一个政企问答大模型项目，请你帮我描述一下项目中用到RAG的实现流程，要求输出详细RAG实现搜索增强流程，包含使用到模型及模型特点，整理成文档，保存在当前目录下，名称为整齐问答RAG实现流程。"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

# 提取并打印响应内容
print(result['choices'][0]['message']['content'])
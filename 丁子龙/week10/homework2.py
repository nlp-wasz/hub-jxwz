import base64
import requests
from openai import OpenAI

image_path = "./2025-11-06_08-19.jpg"
with open(image_path, "rb") as f:
    image_data = f.read()
    base64_str = base64.b64encode(image_data).decode("utf-8")
    image_category = image_path.split('.')[-1]
    data_url = f"data:image/{image_category};base64,{base64_str}"

# 调用模型
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="any"  # Ollama 任意 key 即可
)

response = client.chat.completions.create(
        model="qwen3-vl:8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "你是一个OCR识别专家，请识别该图片中的文字信息，图片中的文本结构可以使用json表示。注意只输出json，不要输出其他内容。"}
                ]
            }
        ],
        temperature=0.7
    )
print(response.choices[0].message.content)

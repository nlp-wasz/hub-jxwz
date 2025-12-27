import base64

import openai

client = openai.OpenAI(
    api_key = "sk-1***",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取并编码图像为base64
with open("./文字截图.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

completion = client.chat.completions.create(
    model= "qwen-vl-plus",
    messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请识别这张图片中的文本内容并格式化输出，不要解释。"
                        }
                    ]
                }
            ],
    max_tokens= 500
)
print(completion.model_dump_json())

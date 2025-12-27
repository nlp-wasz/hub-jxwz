from openai import OpenAI
import base64

# 1. 编码本地图片为Base64（如果API支持直接传URL可跳过此步）
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. 初始化客户端，关键是指定正确的API基地址（base_url）
client = OpenAI(
    api_key="XXXXXX",  # 替换为你的API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 替换为云服务商提供的实际API地址
)

# 3. 构建请求
image_path = "./image/cat01.jpg"
base64_image = encode_image(image_path)

try:
    response = client.chat.completions.create(
        model="qwen3-vl-flash",  # 指定模型名称，根据云服务商提供的模型名填写
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 注意：这里需要根据API的要求组织数据格式，常见的是Base64或直接URL
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请判断这是猫还是狗！"
                    }
                ]
            }
        ],
        max_tokens=1000,  # 控制模型生成的最大长度
    )
    # 4. 提取并打印结果
    print(response.choices[0].message.content)
except Exception as e:
    print(f"调用API时出错: {e}")
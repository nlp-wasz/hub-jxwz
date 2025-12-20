import base64
import requests


def extract_text_from_image(api_key, image_path):
    """
    使用Qwen-VL模型提取图片中的文字
    :param api_key: 阿里云百炼API Key
    :param image_path: 本地图片路径
    :return: 提取的文字内容
    """
    # 1. 将图片编码为Base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # 2. 构造Data URL
    image_data_url = f"data:image/jpeg;base64,{base64_image}"  # 根据实际图片格式调整jpeg/png等

    # 3. 设置API请求
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "qwen-vl-max-latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": "请提取图片中的所有文字内容"
                    }
                ]
            }
        ]
    }

    # 4. 发送请求并返回结果
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# 使用示例
if __name__ == "__main__":
    API_KEY = "sk-24a51bea3ca64b34a8f33e04a9490f44"  # 替换为真实API Key
    IMAGE_PATH = "task02.jpg"  # 替换为图片路径

    text = extract_text_from_image(API_KEY, IMAGE_PATH)
    print("提取的文字：", text)
import json
import requests


def extract_text_from_image(image_url, api_key):
    """
    调用阿里云百炼平台API进行图像中文本提取

    Args:
        image_url (str): 图片URL地址
        api_key (str): 阿里云百炼平台API Key

    Returns:
        str: 提取的文本内容
    """

    # 百炼平台API endpoint
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 请求参数
    payload = {
        # "model": "qwen-vl-plus",
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": image_url
                        },
                        {
                            "text": "请提取图片中的所有文字内容，按照原文格式输出"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "max_tokens": 2048
        }
    }

    # 发送请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        # 提取文本内容
        content = result["output"]["choices"][0]["message"]["content"]

        # 处理不同格式的响应
        if isinstance(content, dict):
            extracted_text = content.get("text", "")
        elif isinstance(content, list):
            first_item = content[0] if content else ""
            if isinstance(first_item, dict):
                extracted_text = first_item.get("text", "")
            else:
                extracted_text = first_item
        else:
            extracted_text = content

        return str(extracted_text).strip()
    else:
        raise Exception(f"API调用失败: {response.status_code} - {response.text}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    # IMAGE_URL = "https://pics0.baidu.com/feed/a9d3fd1f4134970aefe64e3f16fe39c6a6865dc8.jpeg?token=990ba06c5e24cb6ec6157b24db805dcd"  # 替换为实际截图图片URL
    IMAGE_URLS = [
        "https://pics0.baidu.com/feed/a9d3fd1f4134970aefe64e3f16fe39c6a6865dc8.jpeg?token=990ba06c5e24cb6ec6157b24db805dcd",
        "https://pics0.baidu.com/feed/ca1349540923dd54b299245a708cd4d19d8248ab.jpeg?token=4e94e184f4960ac96b6de0045b502e37"
    ]
    API_KEY = "sk-4c44ef4112a04e65910dfdd56774f084"  # 替换为你的API Key

    print("开始提取图像文本...")
    try:
        # 调用API进行文本提取
        for IMAGE_URL in IMAGE_URLS:
            extracted_text = extract_text_from_image(IMAGE_URL, API_KEY)
            print("提取结果:")
            print(extracted_text)
    except Exception as e:
        print(f"文本提取失败: {e}")

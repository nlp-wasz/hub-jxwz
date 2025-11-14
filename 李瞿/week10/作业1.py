import json
import requests


def classify_dog_or_cat(image_url, api_key):
    """
    调用阿里云百炼平台API进行狗猫分类

    Args:
        image_url (str): 图片URL地址
        api_key (str): 阿里云百炼平台API Key

    Returns:
        str: 分类结果 ("dog" 或 "cat")
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
        "model": "qwen-vl-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": image_url
                        },
                        {
                            "text": "这张图片中的动物是狗还是猫？请只回答'dog'或'cat'"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "max_tokens": 10
        }
    }

    # 发送请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        # 提取分类结果 - 适配多种可能的格式
        content = result["output"]["choices"][0]["message"]["content"]

        # 处理内容可能是字典、列表或字符串的情况
        if isinstance(content, dict):
            # 如果是字典，尝试提取文本内容
            classification = content.get("text", "")
        elif isinstance(content, list):
            # 如果是列表，提取第一个元素
            first_item = content[0] if content else ""
            if isinstance(first_item, dict):
                classification = first_item.get("text", "")
            else:
                classification = first_item
        else:
            # 如果是字符串，直接使用
            classification = content

        return str(classification).strip().lower()
    else:
        raise Exception(f"API调用失败: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # 配置参数
    # IMAGE_URL = "https://img1.baidu.com/it/u=1693453968,3801821316&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=1067" # cat
    # IMAGE_URL = "https://img.dingxinwen.cn/image/20250522/e6842f8ea329acfc6617e60ef0abdb92.png"  # dog
    IMAGE_URLS = [
        "https://img1.baidu.com/it/u=1693453968,3801821316&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=1067",  # cat
        "https://img.dingxinwen.cn/image/20250522/e6842f8ea329acfc6617e60ef0abdb92.png",  # dog
    ]

    API_KEY = "sk-4c44ef4112a04e65910dfdd56774f084"  # 替换为你的API Key

    print("开始调用API...")  # 添加调试信息
    try:
        # 调用API进行分类
        for IMAGE_URL in IMAGE_URLS:
            result = classify_dog_or_cat(IMAGE_URL, API_KEY)
            print(f"识别结果: {result}")
    except Exception as e:
        print(f"分类失败: {e}")

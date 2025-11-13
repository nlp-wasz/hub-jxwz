import os
import base64
import requests


def encode_image_to_base64(image_path):
    """
    将本地图片文件编码为Base64字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_image_with_qwen_vl(api_key, image_path, prompt_text):
    """
    使用Qwen-VL模型对图片进行分类
    :param api_key: 阿里云百炼API Key
    :param image_path: 本地图片路径
    :param prompt_text: 分类提示词（例如："dog or cat"）
    :return: 模型返回的分类结果文本
    """
    # 1. 将图片编码为Base64
    base64_image = encode_image_to_base64(image_path)

    # 2. 根据图片格式构造Data URL（需与图片实际格式一致）
    # 支持PNG、JPEG、WEBP等，此处以PNG为例，实际使用需根据图片修改[1](@ref)
    image_data_url = f"data:image/png;base64,{base64_image}"

    # 3. 设置API请求参数
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "qwen-vl-max-latest",  # 使用最新版本的Qwen-VL模型
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}  # 传入Base64格式图片
                    },
                    {
                        "type": "text",
                        "text": prompt_text  # 示例提示词："dog or cat"
                    }
                ]
            }
        ]
    }

    # 4. 发送请求并解析响应
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        classification_result = result["choices"][0]["message"]["content"]
        return classification_result.strip()
    except Exception as e:
        return f"请求失败：{str(e)}"


# 示例使用
if __name__ == "__main__":
    # 配置参数
    DASHSCOPE_API_KEY = "sk-24a51bea3ca64b34a8f33e04a9490f44"  # 需替换为真实API Key[1](@ref)
    IMAGE_PATH = "task01.jpg"  # 替换为本地图片路径
    PROMPT = "dog or cat?"  # 分类提示词

    # 调用函数进行分类
    result = classify_image_with_qwen_vl(DASHSCOPE_API_KEY, IMAGE_PATH, PROMPT)
    print("分类结果:", result)
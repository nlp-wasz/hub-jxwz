# 使用云端qwen-VL模型，完成带文字截图的图，文本的解析转换为文本。
from openai import OpenAI


def extract_text_from_image():
    """
    使用云端qwen-VL模型，完成带文字截图的图，文本的解析转换为文本
    Returns:
        str: 提取的文本内容
    """

    # 创建OpenAI客户端
    client = OpenAI(
        api_key="sk-********************",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 调用模型提取图像中的文本
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":"https://img1.baidu.com/it/u=3362866664,2865770727&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=657"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请识别并提取这张图片中的所有文本内容，按照原文格式输出"
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    # 解析响应结果
    extracted_text = response.choices[0].message.content.strip()
    return extracted_text


# 使用示例
if __name__ == "__main__":
    try:
        result = extract_text_from_image()
        print("提取的文本内容:")
        print(result)
    except Exception as e:
        print(f"错误: {e}")

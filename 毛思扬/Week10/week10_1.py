from openai import OpenAI


def classify_dog_or_cat_openai():
    """
    使用openai库调用云端qwen-VL模型对图像进行分类，识别是dog还是cat
    Returns:
        str: 分类结果 ('dog' 或 'cat')
    """

    # 创建OpenAI客户端
    client = OpenAI(
        api_key="sk-********************",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 调用模型
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://img0.baidu.com/it/u=2923730101,2731579396&fm=253&app=138&f=JPEG?w=500&h=500"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请识别这张图片中的动物是狗还是猫？只需要回答'dog'或'cat'"
                    }
                ]
            }
        ],
        max_tokens=10
    )

    # 解析响应结果
    answer = response.choices[0].message.content.strip().lower()
    return answer


# 使用示例
if __name__ == "__main__":
    try:
        result = classify_dog_or_cat_openai()
        print(f"识别结果: {result}")
    except Exception as e:
        print(f"错误: {e}")

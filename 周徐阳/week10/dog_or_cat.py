from openai import OpenAI
import os


def classify_dog_or_cat(image_url):
    # 设置 API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 调用模型
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": "这张图片中是狗还是猫？请只回答 'dog' 或 'cat'，不要有其他内容。"
                    }
                ]
            }
        ]
    )

    # 解析结果
    result = completion.choices[0].message.content.strip().lower()

    if 'dog' in result or '狗' in result:
        return 'dog'
    elif 'cat' in result or '猫' in result:
        return 'cat'
    else:
        return result


def main():
    # 示例 1: 狗的图片
    dog_image = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"

    print(f"\n测试图片: {dog_image}")
    print("正在识别...")

    try:
        result = classify_dog_or_cat(dog_image)
        print(f"识别结果: {result.upper()}")

        if result == 'dog':
            print("✓ 这是一只狗！")
        elif result == 'cat':
            print("✓ 这是一只猫！")
        else:
            print(f"模型返回: {result}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()

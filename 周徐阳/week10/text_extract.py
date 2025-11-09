from openai import OpenAI
import os


def extract_text_from_image(image_url):

    # 设置 API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 调用模型提取文字
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
                        "text": "请提取这张图片中的所有文字内容，按原文输出，不要添加任何解释。"
                    }
                ]
            }
        ]
    )

    # 返回提取的文字
    return completion.choices[0].message.content.strip()


def main():
    test_image = "https://img-blog.csdnimg.cn/img_convert/3b3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f.png"  # 包含文字的截图

    try:
        # 提取文字
        text = extract_text_from_image(test_image)

        print("\n提取的文字内容：")

    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()

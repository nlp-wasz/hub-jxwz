import openai
import os


class ImageAnalyzer:
    def __init__(self, api_key_env="DASHSCOPE_API_KEY", default_api_key="sk-5ebc25ad675b4a77b1c27549f485f51c"):
        self.client = openai.OpenAI(
            api_key=os.getenv(api_key_env, default_api_key),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def analyze_image(self, image_url, question="这是什么？", enable_thinking=False):
        reasoning_content = ""
        answer_content = ""
        is_answering = False

        completion = self.client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ],
            stream=True,
            extra_body={
                'enable_thinking': enable_thinking,
                "thinking_budget": 81920
            },
        )

        if enable_thinking:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content

        return {
            "reasoning_content": reasoning_content,
            "answer_content": answer_content
        }


def main():
    analyzer = ImageAnalyzer()

    image_url = "http://183.162.196.97:9000/mes/1.jpg"
    question = "这是什么？"

    print(f"正在分析图片: {image_url}")
    print(f"问题: {question}")
    print("-" * 50)

    result = analyzer.analyze_image(image_url, question, enable_thinking=False)

    print("\n" + "=" * 50)
    print("分析完成！")


if __name__ == "__main__":
    main()

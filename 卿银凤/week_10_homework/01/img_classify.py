import requests
import base64
import json
import re


class RobustAnimalClassifier:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def classify_animal(self, image_path, max_retries=3):
        """
        带有重试机制的动物分类
        """
        for attempt in range(max_retries):
            try:
                # 读取图片
                with open(image_path, 'rb') as f:
                    image_data = f.read()

                image_base64 = base64.b64encode(image_data).decode('utf-8')

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                data = {
                    "model": "qwen-vl-plus",
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                                    {"text": "这张图片中的动物是狗还是猫？请只回答'狗'或'猫'。"}
                                ]
                            }
                        ]
                    },
                    "parameters": {"max_tokens": 50, "temperature": 0.1}
                }

                response = requests.post(self.url, headers=headers, json=data, timeout=60)
                response.raise_for_status()

                result = response.json()
                # print(f"API响应: {json.dumps(result, indent=2, ensure_ascii=False)}")  # 调试信息

                # 更健壮的响应解析
                answer = self.extract_answer(result)
                if answer:
                    clean_answer = self.clean_answer(answer)
                    return clean_answer
                else:
                    print(f"尝试 {attempt + 1}: 无法从响应中提取答案")
                    continue  # 重试

            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {str(e)}")
                if attempt == max_retries - 1:
                    return f"最终失败: {str(e)}"

        return "分类失败"

    def extract_answer(self, result):
        """
        从API响应中提取答案文本
        """
        try:
            # 尝试多种可能的响应格式
            if "output" in result and "choices" in result["output"]:
                choice = result["output"]["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

                    # 处理不同格式的内容
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # 如果是列表，提取所有文本内容并合并
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        return " ".join(text_parts) if text_parts else None
                    elif isinstance(content, dict) and "text" in content:
                        return content["text"]

            # 如果以上都不匹配，返回整个响应用于调试
            return str(result)
        except Exception as e:
            print(f"提取答案时出错: {str(e)}")
            return None

    def clean_answer(self, answer):
        """
        清理和标准化模型回答
        """
        if not answer:
            return "无回答"

        # 确保是字符串
        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip().lower()

        if re.search(r'狗|dog', answer):
            return '狗'
        elif re.search(r'猫|cat', answer):
            return '猫'
        else:
            return f"无法确定: {answer}"


# 使用示例
classifier = RobustAnimalClassifier('sk-3a08c7bc652943bba4499dc26d5d2701')
result = classifier.classify_animal('01/animal2.jpg')
print(f"分类结果: {result}")
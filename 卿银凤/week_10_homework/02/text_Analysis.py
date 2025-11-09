import requests
import base64
import json
import re
import os
from PIL import Image
import io


class ScreenshotTextExtractor:
    def __init__(self, api_key):
        """
        初始化截图文本提取器

        Args:
            api_key: 阿里云百炼 API Key
        """
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def image_to_base64(self, image_input):
        """
        将图片转换为 base64 编码

        Args:
            image_input: 图片路径或 PIL Image 对象

        Returns:
            base64 编码的图片字符串
        """
        if isinstance(image_input, str):
            # 从文件路径加载
            with open(image_input, "rb") as image_file:
                image_data = image_file.read()
        elif isinstance(image_input, Image.Image):
            # 已经是 PIL Image 对象
            buffer = io.BytesIO()
            image_input.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
        else:
            raise ValueError("不支持的图片输入类型")

        return base64.b64encode(image_data).decode('utf-8')

    def extract_text_from_screenshot(self, image_input, prompt=None, enhance_quality=True):
        """
        从截图中提取文本内容

        Args:
            image_input: 图片路径或 PIL Image 对象
            prompt: 可选的提示词，默认使用标准文本提取提示
            enhance_quality: 是否使用质量增强模式（多次提取并合并）

        Returns:
            提取的文本内容
        """
        if prompt is None:
            prompt = """请仔细识别这张截图中的所有文字内容，包括：
            1. 所有可见的文本、数字、符号
            2. 界面元素上的标签文字
            3. 按钮、菜单、标题等所有文字
            4. 保持原有的格式和顺序

            请将识别结果以纯文本形式返回，不要添加任何解释或额外内容。"""

        # 准备请求数据
        image_base64 = self.image_to_base64(image_input)

        data = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:image/jpeg;base64,{image_base64}"
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "max_tokens": 1500,  # 增加token限制以容纳更多文本
                "temperature": 0.1  # 低温度以获得更准确的结果
            }
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()

            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]

                # 处理可能的列表格式响应
                if isinstance(content, list):
                    # 提取所有文本部分
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                    content = "\n".join(texts)

                return self.clean_extracted_text(content)
            else:
                return f"API返回格式异常: {result}"

        except requests.exceptions.RequestException as e:
            return f"API请求失败: {str(e)}"
        except Exception as e:
            return f"处理失败: {str(e)}"

    def clean_extracted_text(self, text):
        """
        清理提取的文本，去除多余的标记和格式问题
        """
        # 移除常见的OCR错误标记
        clean_text = text.strip()

        # 处理可能的换行和空格问题
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)  # 多个空行合并为一个
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # 多个空格合并为一个

        return clean_text

    def extract_with_enhancement(self, image_input, strategies=None):
        """
        使用多种策略增强文本提取质量

        Args:
            image_input: 图片路径或 PIL Image 对象
            strategies: 提取策略列表

        Returns:
            增强后的文本提取结果
        """
        if strategies is None:
            strategies = [
                "请提取这张截图中的所有文字，保持原样输出",
                "请识别并转录这张图片中的所有文本内容，包括界面元素、按钮文字、标题等",
                "请将这张截图转换为纯文本格式，包含所有可见文字"
            ]

        results = []
        for strategy in strategies:
            result = self.extract_text_from_screenshot(image_input, prompt=strategy, enhance_quality=False)
            if result and not result.startswith("API请求失败") and not result.startswith("处理失败"):
                results.append(result)

        if not results:
            return "所有提取尝试都失败了"

        # 合并结果，优先选择最长的（通常包含最多信息）
        best_result = max(results, key=len)
        return best_result

    def batch_extract(self, image_paths, output_dir=None):
        """
        批量提取多张截图的文本

        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录（可选，用于保存提取的文本）

        Returns:
            提取结果字典 {图片路径: 提取的文本}
        """
        results = {}

        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                print(f"正在处理第 {i + 1}/{len(image_paths)} 张图片: {path}")
                result = self.extract_with_enhancement(path)
                results[path] = result

                # 保存到文件（如果指定了输出目录）
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.splitext(os.path.basename(path))[0] + ".txt"
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"已保存到: {output_path}")
            else:
                results[path] = "文件不存在"

        return results


# 使用示例
def main():
    # 替换为你的实际 API Key
    API_KEY = "sk-3a08c7bc652943bba4499dc26d5d2701"

    # 创建提取器实例
    extractor = ScreenshotTextExtractor(API_KEY)

    # 单张截图提取
    screenshot_path = "screenshot.png"  # 替换为你的截图路径
    result = extractor.extract_with_enhancement(screenshot_path)

    print("=== 提取的文本内容 ===")
    print(result)

    # 保存到文件
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("文本已保存到 extracted_text.txt")

    # 批量提取示例
    # screenshot_paths = ["screenshot1.png", "screenshot2.png", "screenshot3.png"]
    # results = extractor.batch_extract(screenshot_paths, output_dir="extracted_texts")
    # print("批量提取完成!")


if __name__ == "__main__":
    main()
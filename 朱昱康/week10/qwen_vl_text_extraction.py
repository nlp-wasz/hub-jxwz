#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用阿里云百炼API调用Qwen-VL模型进行图像中文本解析
"""

import base64
import dashscope
from dashscope import MultiModalConversation


def image_to_base64(image_path):
    """将图片转换为base64编码"""
    if image_path.startswith('http'):
        return image_path
    else:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text_from_image(image_path, question="请提取图片中的所有文字内容"):
    """使用阿里云百炼API调用Qwen-VL模型提取图片中的文字"""
    # 设置API密钥
    dashscope.api_key = 'api-key' 
    
    # 准备图片
    image_input = image_to_base64(image_path)
    
    # 构建消息
    messages = [{
        "role": "user",
        "content": [
            {"image": image_input},
            {"text": question}
        ]
    }]
    
    # 调用API
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    # 返回结果
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"API调用失败: {response.message}"


def extract_structured_text(image_path):
    """提取图片中的结构化文字内容"""
    question = """请提取图片中的所有文字内容，并按照以下格式返回：
1. 识别所有可见的文字
2. 如果有表格，请按照表格格式返回
3. 如果有标题，请标识标题层级
4. 保持原始文字的格式和换行"""
    
    return extract_text_from_image(image_path, question)


def extract_key_info(image_path):
    """提取图片中的关键信息"""
    question = """请提取图片中的关键信息，包括：
1. 主要标题
2. 重要数据或数字
3. 联系方式
4. 特殊标记或警告"""
    
    return extract_text_from_image(image_path, question)


def translate_text(image_path):
    """翻译图片中的文字"""
    question = """请提取图片中的文字内容，并将其翻译成英文"""
    
    return extract_text_from_image(image_path, question)


def main():
    """主函数，测试图像文字提取功能"""
    # 测试1：提取所有文字
    doc_image_url = "https://res1-cn.c.hihonor.com/data/attachment/forum/202508/04/52244ef615ce98b7c8ae203734c6619dddd36f8e785ffa4646eece1affa08e2b.jpg"
    print(f"正在分析文档图片: {doc_image_url}")
    result = extract_text_from_image(doc_image_url)
    print(f"文字提取结果:\n{result}\n")
    
    # 测试2：提取结构化文字
    print("结构化文字提取:")
    structured_result = extract_structured_text(doc_image_url)
    print(f"{structured_result}\n")
    
    # 测试3：提取关键信息
    print("关键信息提取:")
    key_info_result = extract_key_info(doc_image_url)
    print(f"{key_info_result}\n")
    
    # 测试4：翻译文字
    print("文字翻译:")
    translation_result = translate_text(doc_image_url)
    print(f"{translation_result}\n")


if __name__ == "__main__":
    main()
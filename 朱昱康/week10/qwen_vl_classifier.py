#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用阿里云百炼API调用Qwen-VL模型进行图像分类：狗或猫识别

本脚本演示如何使用阿里云百炼API调用Qwen-VL多模态大模型来识别图片中的动物是狗还是猫。
"""

import requests
import base64
from PIL import Image
import io
import os
import argparse
from dashscope import Generation


def image_to_base64(image_path):
    """
    将图片转换为base64编码
    
    参数:
        image_path: 图片路径或URL
    
    返回:
        base64编码的图片字符串
    """
    if image_path.startswith('http'):
        # 如果是URL，直接返回URL
        return image_path
    else:
        # 如果是本地文件，读取并转换为base64
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def classify_image_api(image_path, question="Is this a dog or a cat?", api_key=None):
    """
    使用阿里云百炼API调用Qwen-VL模型对图片进行分类，判断是狗还是猫
    
    参数:
        image_path: 图片路径或URL
        question: 向模型提出的问题
        api_key: 阿里云百炼API密钥
    
    返回:
        模型的回答
    """
    if api_key is None:
        return "错误：未提供API密钥"
    
    # 准备图片
    image_input = image_to_base64(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": image_input,
                },
                {"text": question},
            ],
        }
    ]
    
    # 调用API
    response = Generation.call(
        model='qwen-vl-plus',
        messages=messages,
        result_format='message',
        api_key=api_key
    )
    
    # 提取结果
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"API调用失败: {response.message}"


def classify_dog_or_cat_api(image_path, api_key=None):
    """
    更精确的分类函数，只返回"dog"或"cat"
    """
    # 使用更精确的提示词
    question = "Please look at this image and identify if the animal shown is a dog or a cat. Answer with only one word: either 'dog' or 'cat'."
    
    result = classify_image_api(image_path, question, api_key)
    
    # 提取关键词
    result_lower = result.lower()
    if "dog" in result_lower:
        return "dog"
    elif "cat" in result_lower:
        return "cat"
    else:
        return "uncertain"


def main():
    """
    主函数，处理命令行参数并执行图像分类
    """
    parser = argparse.ArgumentParser(description='使用阿里云百炼API调用Qwen-VL模型进行图像分类')
    parser.add_argument('--image', type=str, required=True, help='图片路径或URL')
    parser.add_argument('--api-key', type=str, required=True, help='阿里云百炼API密钥')
    parser.add_argument('--question', type=str, default='Is this a dog or a cat?', help='向模型提出的问题')
    parser.add_argument('--simple', action='store_true', help='只返回dog或cat的简单分类')
    
    args = parser.parse_args()
    
    # 验证API密钥
    if not args.api_key or args.api_key == 'your-api-key-here':
        print("错误：请提供有效的阿里云百炼API密钥")
        print("获取API密钥：https://bailian.console.aliyun.com/")
        return
    
    # 验证图片路径
    if not args.image.startswith('http') and not os.path.exists(args.image):
        print(f"错误：图片文件不存在: {args.image}")
        return
    
    print(f"正在分析图片: {args.image}")
    
    # 执行分类
    if args.simple:
        result = classify_dog_or_cat_api(args.image, args.api_key)
        print(f"分类结果: {result}")
    else:
        result = classify_image_api(args.image, args.question, args.api_key)
        print(f"模型识别结果: {result}")


if __name__ == "__main__":
    main()
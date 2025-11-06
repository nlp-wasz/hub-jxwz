#!/usr/bin/env python3
# ocr_image.py
import os
import base64
import mimetypes
import argparse
import dotenv
import dashscope
from dashscope import MultiModalConversation

dotenv.load_dotenv()

def image_to_base64(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def build_message(image_source: str) -> dict:
    if os.path.isfile(image_source):
        image_url = image_to_base64(image_source)
    else:
        image_url = image_source

    return {
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": "请识别并提取图中所有文字，按原顺序输出，不要添加额外解释。"}
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 图片文字提取")
    parser.add_argument("--image", required=True, help="本地路径或 http(s) 图片地址")
    parser.add_argument("--model", default="qwen3-vl-plus", help="模型名，默认 qwen3-vl-plus")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY 或写入 .env 文件")

    messages = [build_message(args.image)]

    response = MultiModalConversation.call(
        api_key=api_key,
        model=args.model,
        messages=messages,
        stream=True,
        enable_thinking=False
    )

    print("=== OCR 结果 ===")
    for chunk in response:
        text = chunk.output.choices[0].message.get("content", [])
        if text:
            print(text[0]["text"], end="")
    print()

if __name__ == "__main__":
    main()

# python qwen-vl_ocr_image.py --image ./pic/manualtabel.png
# 识别结果：
# === OCR 结果 ===
# 教学进度安排表
# 周次    起讫日期        教学内容
# 1       2.13—2.14       位置与方向（一）
# 2       2.17—2.21       口算除法，笔算除法
# 3       2.24—2.28       有关0的除法，解决问题
# 4       3.03—3.07       整理和复习
# 5       3.10—3.14       复式统计表，口算乘法
# 6       3.17—3.21       笔算乘法
# 7       3.24—3.28       解决问题（连乘）
# 8       3.31—4.03       解决问题（连除）
# 9       4.07—4.11       整理与复习
# 10      4.14—4.18       期中复习
# 11      4.21—4.25       期中测试
# 12      4.28—4.30       面积和面积单位、面积的计算
# 13      5.06—5.09       面积单位间的进率，解决问题
# 14      5.12—5.16       年、月、日，24时计时法
# 15      5.19—5.23       解决问题
# 16      5.26—5.30       认识小数
# 17      6.03—6.06       简单的小数加减法
# 18      6.09—6.13       排列问题、搭配问题
# 19      6.16—6.20       组合问题
# 20      6.23—6.27       总复习
# 21      6.30—7.04       期末考试
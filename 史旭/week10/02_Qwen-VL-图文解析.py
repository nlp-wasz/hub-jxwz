# 使用 Qwen-VL 模型完成图文解析，将图片上的文字解析成文本内容
# 使用 本地Qwen-vl 和 远程Qwen-VL API  完成图文分类任务
import json
import os

import torch, requests, base64
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI


# 1.本地Qwen-VL模型使用
def localhost_qwen_vl(img_path, text):
    # 1.加载 processor 和 model
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "../../models/Qwen/Qwen2-5-VL-3B-Instruct", use_fast=False, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "../../models/Qwen/Qwen2-5-VL-3B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 2.加载 图片信息
    if img_path.startswith("http"):
        img = Image.open(BytesIO(requests.get(img_path).content)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')

    # 3.构建 messages
    messages = [
        {"role": "system", "content": "你是一个专业的assistant，请根据图文信息回答用户的问题！"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                    "resized_height": 500,
                    "resized_width": 500
                },
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    ]

    # 4.获取提示词模板
    prompt_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 5.获取 图片、音视频 处理信息
    image_inputs, videos_inputs = process_vision_info(messages)

    # 6.对 文本、图片、音视频 进行预处理
    inputs = processor(
        images=image_inputs,
        text=[prompt_template],
        videos=videos_inputs,
        return_tensors="pt"
    ).to("cuda")

    # 7.模型生成回答
    with torch.no_grad():
        generators = model.generate(**inputs, max_new_tokens=128)

    generators_encoder = [generator[len(input_ids):] for input_ids, generator in zip(inputs["input_ids"], generators)]
    generator_decoder = processor.batch_decode(generators_encoder, skip_special_tokens=True,
                                               clean_up_tokenization_space=True)

    print(" ".join(generator_decoder))


# 2.调用 Qwen-VL API 完成图文分类任务
def api_qwen_vl(img_path, text):
    if img_path.startswith("http"):
        buffered = BytesIO(requests.get(img_path).content)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img = Image.open(img_path).convert('RGB')
        # 转换为 BytesIO 流
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    messages = [
        {"role": "system", "content": "你是一个专业的assistant，请根据图文信息回答用户的问题！"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_base64}"
                },
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    ]

    def openai():
        # 1.创建 openai
        qwen_vl = OpenAI(
            api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 2.调用chat模块
        res = qwen_vl.chat.completions.create(
            model="qwen3-vl-plus",
            messages=messages
        )

        print(res.choices[0].message.content)
        print("=" * 100)

    def langchain_chatopenai():
        # 1.创建 ChatOpenAI
        qwen_vl = ChatOpenAI(
            api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-vl-plus"
        )

        # 2.执行 invoke
        res = qwen_vl.invoke(messages)

        print(res.model_dump_json(indent=2))
        print("=" * 100)

    def langchain_init_chat_model():
        # 1.加载配置信息
        import dotenv
        dotenv.load_dotenv("../../week08-智能体Agent基础/keyANDurl.env")

        # 2.创建 模型
        qwen_vl = init_chat_model(model="qwen3-vl-plus", model_provider="openai")

        # 3.执行 invoke
        res = qwen_vl.invoke(messages)

        print(res.model_dump_json(indent=2))
        print("=" * 100)

    openai()
    # langchain_chatopenai()
    # langchain_init_chat_model()


if __name__ == '__main__':
    # 1.本地Qwen-VL模型使用
    img_path = "../data/img_2.png"
    text = "根据行级提取图片上的所有文字，以json格式输出。"
    # localhost_qwen_vl(img_path, text)

    # 2.调用 Qwen-VL API 完成图文分类任务
    api_qwen_vl(img_path, text)

# 使用 多模态模型（CLIP，Qwen-VL）等完成图文问答、OCR、实体识别等任务
import base64

import requests, numpy as np, pandas as pd, torch
from openai import OpenAI
from sklearn.preprocessing import normalize

from transformers import ChineseCLIPProcessor, ChineseCLIPModel, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info


# 1.CLIP 多模态模型使用
# 步骤①：Processor 处理器对 图片 | 文本 等进行处理（图像调整尺寸、归一化等操作，文本分词、padding、truncation等操作）
# 步骤②：CLIPModel 模型获取 图片 | 文本 等特征信息
# 步骤③：手动对 特征信息 进行 normalize 归一化处理，最后计算余弦相似度
def clip_use(image_path, text):
    # 1.加载 CLIP 处理器和模型
    clip_processor = ChineseCLIPProcessor.from_pretrained("../models/clip/chinese-clip-vit-base-patch16")
    clip_model = ChineseCLIPModel.from_pretrained("../models/clip/chinese-clip-vit-base-patch16")

    # 2.图像文本预处理
    if image_path.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image_process = clip_processor(images=image, return_tensors="pt")

    text_process = clip_processor(text=text, return_tensors="pt", padding=True, truncate=True, max_length=512)

    # 3.获取特征编码
    with torch.no_grad():
        image_feature = clip_model.get_image_features(**image_process)
        text_feature = clip_model.get_text_features(**text_process)

        image_feature = normalize(image_feature.data.numpy())
        text_feature = normalize(text_feature.data.numpy())

    # 4.计算相似度
    similator = np.dot(image_feature, text_feature.T)
    similator_dict = {text[idx]: score for idx, score in enumerate(similator[0])}
    similator_sort = sorted(similator_dict.items(), key=lambda item: item[1], reverse=True)
    print(similator_sort)


# 2.Qwen-VL 多模态模型使用
# 步骤①：构建messages，调用 processor.apply_chat_template() 获取prompt 带生成assistant的模板
# 步骤②：调用 qwen_vl_utils.process_vision_info(messages) 获取 图片、音频等信息
# 步骤③：使用 process 处理器对 images、text、videos 等输出进行预处理
# 步骤④：将 process 预处理的输出 交给qwen-vl多模态模型，调用 model.generator()生成回答
def qwen_vl_use(image_path, text):
    # 1. 加载 Qwen-VL 处理器和模型
    qwen_vl_process = Qwen2_5_VLProcessor.from_pretrained(
        "../models/Qwen/Qwen2-5-VL-3B-Instruct", use_fast=False, trust_remote_code=True)
    qwen_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "../models/Qwen/Qwen2-5-VL-3B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,  # 开启4-bit量化，减少GPU使用，防止OOM
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 2. 构建 messages
    if image_path.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": 500,
                    "resized_width": 100,
                },
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    ]

    # 3.processor.apply_chat_template() 获取prompt模板
    prompt_template = qwen_vl_process.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4.qwen_vl_utils.process_vision_info(messages) 提取图片、音视频信息
    image_inputs, videos_inputs = process_vision_info(messages)

    # 5.processor 对图片、文本、音视频进行预处理（图片像素处理，文本分词处理等）
    inputs = qwen_vl_process(
        images=image_inputs,
        text=[prompt_template],
        videos=videos_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # 6.模型生成回答
    with torch.no_grad():
        generators = qwen_vl_model.generate(**inputs, max_new_tokens=1024)

    # 7.提取回答
    generator_encoder = [generator[len(input_ids):] for input_ids, generator in zip(inputs["input_ids"], generators)]
    generator_decoder = qwen_vl_process.batch_decode(generator_encoder)

    print(generator_decoder)


# 3.调用 Qwen-VL多模态模型  API模式
def qwen_vl_api_use(image_path, text):
    from openai import OpenAI
    import base64

    # 1.创建 openai 连接对象
    qwen_vl = OpenAI(
        api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 2.构建 messages（image_url：url 或者 base64编码后的 BytesIO字节流）
    if image_path.startswith("http"):
        img_bytes = BytesIO(requests.get(image_path).content)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    else:
        img = Image.open(image_path)
        buffered = BytesIO()
        img.save(buffered, format="jpeg")
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

    # 3.调用模型
    res = qwen_vl.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages
    )

    print(res.choices[0].message.content)


# 4.提取 pdf 文件中的 tent、表格、image（然后交给 qwen-vl 多模态视觉模型进行处理）
def extract_pdf_image(pdf_path):
    import pdfplumber  # 只能提取pdf页面文字 这类简单操作
    import pymupdf as fitz  # 更强大的pdf处理模块

    # 1.打开 pdf 文件
    pdf = fitz.open(pdf_path)

    # 2.加载 第一页（或者 循环）
    page_1 = pdf.load_page(0)
    # for page in pdf:
    #     break

    # 3.提取 当前页 文本内容
    text = page_1.get_text()
    print(text)

    # 4.提取 图片
    # full = True，如果图片在两页，也可以提取
    images = page_1.get_images(full=True)

    # 循环 图像列表信息，对每一张图片进行单独处理
    for image in images:
        # 1.image 包含多种信息 (img_xref, smask_xref, width, height, bpc, colorspace, alt_img_name, img_filter, img_ref_count)
        # 只关注 xref（图片ID，后续根据ID提取图片信息）
        img_xref = image[0]

        # 2.用 pdf.extract_image(img_xref) 获取图像数据字典
        image_info = pdf.extract_image(img_xref)

        # 3.从 image_info 中获取 image原始字节 和 ext后缀信息
        image_bytes = image_info["image"]
        ext = "jpeg" if image_info["ext"] == "jpg" else image_info["ext"]

        # 3.创建 openai 连接对象
        qwen_vl = OpenAI(
            api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 转换为 base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": "你是一个专业的assistant，请根据图文信息回答用户的问题！"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{ext};base64,{image_base64}"
                    },
                    {
                        "type": "text",
                        "text": "描述图片内容"
                    }
                ]
            }
        ]

        # 3.调用模型
        res = qwen_vl.chat.completions.create(
            model="qwen3-vl-plus",
            messages=messages
        )

        print(res.choices[0].message.content)
        break


if __name__ == "__main__":
    # 1.CLIP 多模态模型使用
    # image_path = "./data/img/00b7e44b872ce5ad0be55f9564c77e29625250a4.jpg"
    # text = ["运动服", "运动场", "踢足球", "两个", "短发", "高尔夫", "戴着帽子", "女人", "拖鞋"]
    # clip_use(image_path, text)

    # 2.Qwen-VL 多模态模型使用
    # image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    # text = "Describe this image."
    # qwen_vl_use(image_path, text)

    # 3.调用 Qwen-VL多模态模型  API模式
    # image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    # text = "Describe this image."
    # qwen_vl_api_use(image_path, text)

    # 4.提取 pdf 文件中的 tent、表格、image（然后交给 qwen-vl 多模态视觉模型进行处理）
    pdf_path = "./data/dPy1eeivs4.pdf"
    extract_pdf_image(pdf_path)

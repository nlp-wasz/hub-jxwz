# 基础库
import time
import traceback
import torch
import numpy as np
from fastapi import FastAPI

# 自定义模块
from data_schema import TextClassifyResponse, TextClassifyRequest
from model.bert import model_for_bert  # 复用原bert.py的函数
from logger import logger


# -------------------------- 1. 启动时预加载模型/分词器（仅1次） --------------------------
# 全局变量：存储推理所需工具（避免每次请求重复训练/加载）
TOKENIZER = None
MODEL = None
LABEL_ENCODER = None

# 预加载逻辑（启动服务时执行）
try:
    # 调用原model_for_bert函数（训练+返回工具），仅首次运行时训练，后续加载缓存
    TOKENIZER, MODEL, LABEL_ENCODER = model_for_bert()
    MODEL.eval()  # 切换为推理模式（关闭训练层）
    MODEL.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 适配GPU/CPU
    # 修复：删除emoji符号，用纯文本描述日志
    logger.info("BERT模型、分词器、标签编码器预加载完成")
except Exception as e:
    # 修复：删除emoji符号
    logger.error(f"预加载失败：{str(e)}")
    logger.error(traceback.format_exc())


# -------------------------- 2. 初始化FastAPI --------------------------
app = FastAPI(title="BERT文本分类接口", version="1.0")


# -------------------------- 3. 核心分类接口 --------------------------
@app.post("/v1/text-cls/bert", response_model=TextClassifyResponse)
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    start_time = time.time()
    # 初始化响应
    resp = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg="ok"
    )

    try:
        # 检查预加载是否成功
        if not all([TOKENIZER, MODEL, LABEL_ENCODER]):
            raise RuntimeError("模型未加载，无法处理请求")

        # 1. 文本编码（适配模型输入）
        encoding = TOKENIZER(
            req.request_text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(next(MODEL.parameters()).device)  # 与模型同设备

        # 2. 模型推理（关闭梯度计算，加速）
        with torch.no_grad():
            logits = MODEL(**encoding).logits
            num_classes = len(LABEL_ENCODER.classes_)
            # 适配单/多类别
            if num_classes == 1:
                pred_id = 1 if torch.sigmoid(logits).item() > 0.5 else 0
            else:
                pred_id = torch.argmax(logits, dim=-1).item()

        # 3. 转换为原始标签
        resp.classify_result = LABEL_ENCODER.inverse_transform([pred_id])[0]
        # 修复：删除emoji符号
        logger.info(f"请求{req.request_id}：分类成功，结果={resp.classify_result}")

    except Exception as e:
        resp.error_msg = f"分类失败：{str(e)}"
        resp.classify_result = ""
        logger.error(f"请求{req.request_id}：{resp.error_msg}")
        logger.error(traceback.format_exc())

    # 计算耗时
    resp.classify_time = round(time.time() - start_time, 3)
    return resp
from fastapi import FastAPI
import time
import traceback
from .schemas import TextClassifyRequest, TextClassifyResponse
from model.bert_inference import BertClassifier
import logging
import os


# 方法 ：使用模块运行方式
# 在项目根目录运行：
#
# python -m api.main

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT文本分类API",
    description="使用微调的BERT模型进行文本情感分析",
    version="1.0"
)


# 加载模型
def load_model():
    try:
        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            current_dir,
            "..",
            "assets",
            "models",
            "bert-finetuned"  # 微调后的模型
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        classifier = BertClassifier(model_path)
        logger.info("模型加载成功")
        return classifier
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return None


# 加载模型
classifier = load_model()


@app.get("/")
def root():
    return {
        "message": "BERT文本分类API服务已启动",
        "endpoints": {
            "classify": "/v1/text-cls/bert",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.post("/v1/text-cls/bert", response_model=TextClassifyResponse)
def bert_classify(req: TextClassifyRequest):
    """
    使用微调的BERT模型进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0.0,
        error_msg=""
    )

    try:
        if classifier is None:
            raise RuntimeError("模型未加载成功")

        response.classify_result = classifier.predict(req.request_text)
        response.error_msg = "ok"
        logger.info(f"请求 {req.request_id} 处理成功")
    except Exception as e:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()
        logger.error(f"请求 {req.request_id} 处理失败: {str(e)}")

    response.classify_time = round(time.time() - start_time, 3)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
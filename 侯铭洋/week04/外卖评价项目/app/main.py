from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# 导入自定义模块
from .models import load_model, predict_sentiment
from .schemas import SentimentRequest, SentimentResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title="外卖评价情感分析 API",
    description="使用微调的BERT模型对外卖评价进行情感分析",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 文档地址
    redoc_url="/redoc",  # ReDoc 文档地址
)

# 添加 CORS 中间件，允许前端应用调用API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该更严格限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量，用于缓存模型和分词器
model = None
tokenizer = None


@app.on_event("startup")
async def startup_event():
    """
    应用启动时自动执行，用于加载模型
    """
    global model, tokenizer
    try:
        # 获取模型路径（假设模型保存在项目根目录的 saved_models 文件夹中）
        model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "fine-tuned-bert")
        model, tokenizer = load_model(model_path)
        logger.info("应用启动完成，模型已加载")
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        # 这里可以选择让应用启动失败，或者继续运行但标记服务不可用
        raise e


@app.get("/")
async def root():
    """
    根端点，返回服务基本信息
    """
    return {
        "message": "外卖评价情感分析 API",
        "status": "运行中",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点，用于监控服务状态
    """
    if model is not None and tokenizer is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """
    情感分析预测端点

    - **text**: 需要分析情感的文本
    """
    try:
        # 调用预测函数
        result = predict_sentiment(request.text)

        # 返回预测结果（会自动根据SentimentResponse模型进行验证）
        return result
    except Exception as e:
        logger.error(f"预测请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 可选：添加一个批量预测端点
@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """
    批量情感分析预测端点

    - **texts**: 需要分析情感的文本列表
    """
    try:
        results = []
        for text in texts:
            result = predict_sentiment(text)
            results.append(result)
        return {"results": results}
    except Exception as e:
        logger.error(f"批量预测请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 可选：添加模型信息端点
@app.get("/model-info")
async def model_info():
    """
    返回当前加载的模型信息
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    return {
        "model_type": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "num_parameters": sum(p.numel() for p in model.parameters())
    }
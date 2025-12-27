from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义请求模型
class SentimentRequest(BaseModel):
    text: str


# 定义响应模型
class SentimentResponse(BaseModel):
    sentiment: int
    confidence: float
    positive_prob: float
    negative_prob: float
    processing_time: float


# 加载模型和tokenizer
model_path = './sentiment_model'
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise e

app = FastAPI(title="外卖评价情感分析API", version="1.0")


@app.get("/")
async def root():
    return {"message": "外卖评价情感分析API"}


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    start_time = time.time()

    try:
        # 文本预处理和编码
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        # 模型预测
        with torch.no_grad():
            outputs = model(**inputs)

        # 计算概率
        probs = F.softmax(outputs.logits, dim=1)
        positive_prob = probs[0][1].item()
        negative_prob = probs[0][0].item()

        # 确定情感
        sentiment = 1 if positive_prob > negative_prob else 0
        confidence = max(positive_prob, negative_prob)

        processing_time = time.time() - start_time

        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"预测出错: {e}")
        raise HTTPException(status_code=500, detail=f"预测出错: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

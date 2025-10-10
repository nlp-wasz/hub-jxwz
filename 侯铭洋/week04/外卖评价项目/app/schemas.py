from pydantic import BaseModel, Field
from typing import Optional


# 定义请求体模型
class SentimentRequest(BaseModel):
    text: str = Field(..., example="这家餐厅的菜品非常美味，送货也很快", description="需要分析情感倾向的文本")

    # Pydantic 配置示例（可选）
    class Config:
        schema_extra = {
            "example": {
                "text": "这家餐厅的菜品非常美味，送货也很快"
            }
        }


# 定义响应体模型
class SentimentResponse(BaseModel):
    sentiment: int = Field(..., example=1, description="情感标签 (0: 负面, 1: 正面)")
    confidence: float = Field(..., example=0.987, description="预测置信度 (0-1之间)")
    sentiment_label: str = Field(..., example="正面", description="情感标签的文字描述")

    # 可选：添加每个类别的概率
    probabilities: Optional[dict] = Field(
        None,
        example={"负面": 0.013, "正面": 0.987},
        description="每个类别的预测概率"
    )

    class Config:
        schema_extra = {
            "example": {
                "sentiment": 1,
                "confidence": 0.987,
                "sentiment_label": "正面",
                "probabilities": {"负面": 0.013, "正面": 0.987}
            }
        }
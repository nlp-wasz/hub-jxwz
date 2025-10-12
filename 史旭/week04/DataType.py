from pydantic import BaseModel, Field
from typing import List, Union, Optional


# 请求体
class Request(BaseModel):
    epochs: int = Field(..., description="模型训练/预测轮次【3-4】")
    data_size: int = Field(..., description="模型数据集样本数量【<=500】")


class RequestPredict(BaseModel):
    new_text: str = Field(..., description="需要预测的文本信息")


class Response(BaseModel):
    train_loss: List[List[float]] = Field(..., description="模型每个训练轮次的损失值变化情况")
    test_precision: List[str] = Field(..., description="模型每个预测轮次的精度")
    run_time: Optional[str] = Field(..., description="耗时")

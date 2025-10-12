from pydantic import BaseModel, Field
from typing import List, Optional, Union

class EvaluationClassifyRequest(BaseModel):
    """
    评价分类请求模型
    """
    request_id: Optional[str] = Field(..., description = "请求id")
    request_evaluation: Union[str, List[str]] = Field(..., description = "请求文本、字符串或列表")

class EvaluationClassifyResponse(BaseModel):
    """
    评价分类响应模型
    """
    request_id: Optional[str] = Field(..., description = "请求id")
    request_evaluation: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description = "分类结果")
    classify_time: float = Field(..., description = "分类耗时")
    error_message: str = Field(..., description = "异常信息")

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from typing_extensions import Literal

class ExtractionRequest(BaseModel):
    text: str = Field(..., description="需要抽取信息的文本")

class PromptExtractionResponse(BaseModel):
    domain: str = Field(..., description="领域类别")
    intent: str = Field(..., description="意图类别")
    slots: Dict[str, Any] = Field(..., description="实体识别结果")

class ToolExtractionResponse(BaseModel):
    domain: Literal["music", "app", "news", "bus"] = Field(..., description="领域")
    intent: Literal["OPEN", "SEARCH", "ROUTE", "QUERY"] = Field(..., description="意图")
    Src: Optional[str] = Field(None, description="出发地")
    Des: Optional[str] = Field(None, description="目的地")

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    model: str = Field(..., description="使用的模型")
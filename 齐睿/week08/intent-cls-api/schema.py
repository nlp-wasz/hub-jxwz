from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import Dict, Any

class IntentClass(BaseModel):
    """意图识别"""

    text: str = Field(description="用户输入的原始文本，即待识别意图的文本。", default="")
    domain: str = Field(description="领域类别，按照给出的类别进行识别和划分，给出所属的领域。")
    intent: str = Field(description="意图类别，按照给出的类别进行识别和划分，给出所属的意图。")
    slots: Dict[str, Any] = Field(
        description="实体插槽，将原始文本中的实体提取出来，一般以\"name\": \"XXXX\"形式存放,XXXX代表原始实体文本")  # 动态槽位参数

    class Config:
        # 允许额外字段但会触发验证警告
        extra = "allow"

class IntentClassifyRequest(BaseModel):
    """意图识别请求数据模型"""

    text: str = Field(description="待分类的文本")
    class Config:
        # 允许额外字段但会触发验证警告
        extra = "allow"

class IntentClassifyResponse(BaseModel):
    domain: str = Field(description="领域类别，按照给出的类别进行识别和划分，给出所属的领域。")
    intent: str = Field(description="意图类别，按照给出的类别进行识别和划分，给出所属的意图。")
    slots: Dict[str, Any] = Field(
        description="实体插槽，将原始文本中的实体提取出来，一般以\"name\": \"XXXX\"形式存放,XXXX代表原始实体文本")  # 动态槽位参数

    class Config:
        # 允许额外字段但会触发验证警告
        extra = "allow"
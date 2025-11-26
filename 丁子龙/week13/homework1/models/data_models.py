from datetime import datetime
from typing import Optional, List, Dict, Literal, Union, Any
from pydantic import BaseModel, Field, conint

class User(BaseModel):
    user_id: int
    user_name: str
    user_role: str
    register_time: datetime
    status: bool

class BasicResponse(BaseModel):
    code: int
    message: str
    data : Optional[Union[List[Any], Any]]

class RequestForUserLogin(BaseModel):
    user_name: str
    password: str

class RequestForUserRegister(BaseModel):
    user_name: str
    password: str
    user_role: str

class RequestForUserResetPassword(BaseModel):
    user_name: str
    password: str
    new_password: str

class RequestForUserChangeInfo(BaseModel):
    user_name: str
    user_role: Optional[str]
    status: Optional[bool]

# 用户在对话，传入的信息
class RequestForChat(BaseModel):
    content: str = Field(..., description="用户的提问")
    user_name: str = Field(..., description="用户名")
    session_id: Optional[str] = Field(None, description="对话session_id, 获取对话上下文")
    task: Optional[str] = Field(None, description="对话任务")
    tools: Optional[List[str]] = Field(None, description="可选的工具列表")

    # 后序可以持续增加，用户输入图、上传文件、链接、音频、视频，复杂的解析
    image_content: Optional[str] = Field(None)
    file_content: Optional[str] = Field(None)
    url_content: Optional[str] = Field(None)
    audio_content: Optional[str] = Field(None)
    video_content: Optional[str] = Field(None)

    # 后序可以持续增加，对话模型
    vison_mode: Optional[bool] = Field(False)
    deepsearch_mode: Optional[bool] = Field(False)
    sql_interpreter: Optional[bool] = Field(False)
    code_interpreter: Optional[bool] = Field(False)

class ResponseForChat(BaseModel):
    response_text: str
    session_id: Optional[str] = Field(None)
    response_code: Optional[str] =  Field(None)
    response_sql: Optional[str] =  Field(None)

class StockFavInfo(BaseModel):
    stock_code: str
    create_time: datetime


class ChatSession(BaseModel):
    user_id: int
    session_id: str
    title: str
    start_time: datetime
    feedback: Optional[bool] = Field(None)
    feedback_time: Optional[datetime] = Field(None)
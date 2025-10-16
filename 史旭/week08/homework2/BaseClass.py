# FaseApi 请求和相应 模板类
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Literal

a = ["你去玩手机了吧用英语怎么说", " 分手时，背一首诗吧？ ", " 本期七星彩的中奖号码是多少？ ", " 何以解忧的下一句是什么？ ",
     " 请帮我调频90.2连云港经济广播电台"]


class Request(BaseModel):
    request_id: str = Field(..., description="请求ID")
    request_content: Union[List[str], str] = Field(default=a, description="请求内容")


class Response(BaseModel):
    request_id: str = Field(..., description="请求ID")
    request_content: Union[List[str], str] = Field(..., description="请求内容")

    response_res: Union[str, List[str], List[Dict[str, object]], Dict[str, object]] = Field(..., description="响应结果")
    response_time: str = Field(..., description="响应时间")
    response_message: str = Field(..., description="响应信息（error，success）")

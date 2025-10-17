from fastapi import FastAPI
from usingTools import usingTools
from usingPrompt import few_shot_prompting,zero_shot_prompting
from pydantic import BaseModel, Field
from typing import List,Dict,Any,Optional,Union
app = FastAPI()


class TextClassifyRequest(BaseModel):
    request_id: str = Field(description="请求id")
    request_text: str = Field(description="请求文本")

class TextClassifyResponse(BaseModel):
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str], Dict[str, Any]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")


@app.post("/v1/few_shot_prompting")
def few_shot_prompting1(req: TextClassifyRequest) -> TextClassifyResponse:
    result = TextClassifyResponse(request_id = req.request_id,
                                  request_text = req.request_text,
                                  classify_result = "",
                                  classify_time= 0,
                                  error_msg= "")
    result.classify_result = few_shot_prompting(req.request_text)
    result.classify_time = 1
    result.error_msg = "ok"
    return result

@app.post("/v1/zero_shot_prompting")
def zero_shot_prompting1(req: TextClassifyRequest) -> TextClassifyResponse:
    result = TextClassifyResponse(request_id=req.request_id,
                                  request_text=req.request_text,
                                  classify_result="",
                                  classify_time=0,
                                  error_msg="")
    result.classify_result = zero_shot_prompting(req.request_text)
    result.classify_time = 1
    result.error_msg = "ok"
    return result

@app.post("/v1/usingTools")
def usingTools1(req: TextClassifyRequest) -> TextClassifyResponse:
    result = TextClassifyResponse(request_id=req.request_id,
                                  request_text=req.request_text,
                                  classify_result="",
                                  classify_time=0,
                                  error_msg="")
    result.classify_result = usingTools(req.request_text)
    result.classify_time = 1
    result.error_msg = "ok"
    return result


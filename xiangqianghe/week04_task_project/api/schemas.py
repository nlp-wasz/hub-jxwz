from pydantic import BaseModel

class TextClassifyRequest(BaseModel):
    request_id: str
    request_text: str

class TextClassifyResponse(BaseModel):
    request_id: str
    request_text: str
    classify_result: str
    classify_time: float
    error_msg: str
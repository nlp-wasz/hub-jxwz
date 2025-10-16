"""
调用llm02
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 导入识别函数
from llm02 import recognize

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/recognize")
def recognize_text(request: TextRequest):
    """识别接口"""
    result = recognize(request.text)
    return result.model_dump() if result else {"error": "识别失败"}


@app.get("/")
def home():
    return {"message": "联合识别 API", "endpoint": "POST /recognize"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
调用提示词llm
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 导入三种识别方法
from llm01 import (
    joint_recognition,
    joint_recognition_with_examples,
    joint_recognition_structured
)
import json

app = FastAPI()


class TextRequest(BaseModel):
    text: str
    method: str = "basic"  # basic / few-shot / structured


@app.post("/recognize")
def recognize_text(request: TextRequest):
    """
    识别接口
    method 可选: basic, few-shot, structured
    """
    try:
        # 根据方法选择不同的识别函数
        if request.method == "few-shot":
            result_text = joint_recognition_with_examples(request.text)
        elif request.method == "structured":
            result_text = joint_recognition_structured(request.text)
        else:  # basic
            result_text = joint_recognition(request.text)

        # 尝试解析 JSON
        try:
            result = json.loads(result_text)
            return result
        except:
            # 如果不是 JSON，返回原始文本
            return {"result": result_text}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {
        "message": "联合识别 API (Prompt Engineering)",
        "endpoint": "POST /recognize",
        "methods": ["basic", "few-shot", "structured"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

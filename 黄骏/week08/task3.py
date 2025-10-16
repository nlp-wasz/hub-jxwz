import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tast2 import extract_ner, NerTask
from typing import Optional

app = FastAPI()

class Request(BaseModel):
    user_prompt: str = Field(..., example="请帮我打开uc")

@app.get("/")
def health_check():
    return {"status": "正常", "message": "NER服务运行中"}

@app.post("/extract")
async def extract(request: Request) -> Optional[NerTask]:
    """调用extract_ner函数提取领域、意图和实体"""
    try:
        if not request.user_prompt.strip():
            raise HTTPException(status_code=400, detail="输入不能为空")
        result = await extract_ner(request.user_prompt)
        if not result:
            raise HTTPException(status_code=500, detail="提取失败")
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
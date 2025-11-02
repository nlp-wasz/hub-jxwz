from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ExtractionRequest, ExtractionResponse
from agents import ExtractionAgent
import uvicorn

app = FastAPI(
    title="NLU信息抽取服务",
    description="领域识别、意图识别、实体识别服务",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局智能体实例
agent = ExtractionAgent()


@app.get("/")
async def root():
    return {"message": "NLU信息抽取服务正常运行"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_intent(request: ExtractionRequest):
    """信息抽取主接口"""
    try:
        result = agent.call(request.text, request.method)

        if result:
            return ExtractionResponse(
                success=True,
                data=result,
                method=request.method
            )
        else:
            return ExtractionResponse(
                success=False,
                error="信息抽取失败",
                method=request.method
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")



@app.post("/extract/batch")
async def batch_extract(texts: list[str]):
    """批量处理接口"""
    results = []
    for text in texts:
        result = agent.call(text)
        results.append({
            "text": text,
            "result": result.model_dump() if result else None
        })
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    ExtractionRequest,
    PromptExtractionResponse,
    ToolExtractionResponse,
    HealthResponse
)
from extractors import extraction_agent
import uvicorn

app = FastAPI(
    title="信息抽取API服务",
    description="基于阿里云百炼平台的信息抽取服务，支持提示词和工具两种抽取方法",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="服务运行正常",
        model="qwen-plus"
    )


@app.post("/extract/prompt", response_model=PromptExtractionResponse)
async def extract_with_prompt(request: ExtractionRequest):
    """
    使用提示词方法进行信息抽取
    """
    try:
        result = extraction_agent.extract_with_prompt(request.text)
        return PromptExtractionResponse(
            domain=result.get("domain", "unknown"),
            intent=result.get("intent", "unknown"),
            slots=result.get("slots", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")


@app.post("/extract/tool", response_model=ToolExtractionResponse)
async def extract_with_tool(request: ExtractionRequest):
    """
    使用tool进行信息抽取
    """
    try:
        result = extraction_agent.extract_with_tools(request.text)
        if result is None:
            raise HTTPException(status_code=500, detail="工具抽取失败")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")


@app.post("/extract/both")
async def extract_both_methods(request: ExtractionRequest):
    """
    同时使用两种方法进行信息抽取并返回对比结果
    """
    try:
        prompt_result = extraction_agent.extract_with_prompt(request.text)
        tool_result = extraction_agent.extract_with_tools(request.text)

        return {
            "text": request.text,
            "prompt_method": prompt_result,
            "tool_method": tool_result.dict() if tool_result else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
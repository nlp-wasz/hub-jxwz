import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from config.settings import settings
from src.extractors import PromptEngineeringExtractor, AgentBasedExtractor

# 创建FastAPI应用
app = FastAPI(
    title="LLM文本信息抽取API",
    description="基于大型语言模型的文本信息抽取服务，支持领域识别、意图识别和实体识别",
    version="1.0.0"
)

# 定义请求模型
class ExtractionRequest(BaseModel):
    """文本信息抽取请求模型"""
    text: str = Field(..., description="需要抽取信息的文本")
    method: Literal["prompt_engineering", "agent_based"] = Field(
        default="prompt_engineering", 
        description="抽取方法：prompt_engineering（提示词工程）或agent_based（智能体）"
    )

# 定义响应模型
class ExtractionResponse(BaseModel):
    """文本信息抽取响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="抽取结果数据")
    method: str = Field(..., description="使用的抽取方法")

# 初始化抽取器
prompt_extractor = PromptEngineeringExtractor()
agent_extractor = AgentBasedExtractor()

# 定义根路径
@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用LLM文本信息抽取API",
        "version": "1.0.0",
        "model": settings.default_model,
        "methods": ["prompt_engineering", "agent_based"],
        "endpoints": ["/extract", "/docs"]
    }

# 定义抽取接口
@app.post("/extract", response_model=ExtractionResponse)
async def extract_information(request: ExtractionRequest):
    """
    文本信息抽取接口
    
    根据选择的抽取方法，从输入文本中抽取领域类别、意图类型和实体标签
    """
    try:
        # 根据方法选择抽取器
        if request.method == "prompt_engineering":
            extractor = prompt_extractor
        elif request.method == "agent_based":
            extractor = agent_extractor
        else:
            raise HTTPException(status_code=400, detail="不支持的抽取方法")
        
        # 执行抽取
        result = extractor.extract(request.text)
        
        # 转换结果格式
        if request.method == "agent_based" and result:
            result = result.model_dump()
        
        # 返回响应
        return ExtractionResponse(
            success=True,
            message="信息抽取成功",
            data=result,
            method=request.method
        )
        
    except Exception as e:
        # 返回错误响应
        return ExtractionResponse(
            success=False,
            message=f"信息抽取失败: {str(e)}",
            data=None,
            method=request.method
        )

# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "服务运行正常"}

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
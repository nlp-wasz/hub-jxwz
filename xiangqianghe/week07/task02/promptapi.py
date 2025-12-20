from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import openai
import json

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-gcuimbebodlqdoldluiucssywbxlatnoiyfwokewytvayzgt",
    base_url="https://api.siliconflow.cn/v1"
)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="信息抽取API服务",
    description="基于大模型的专业信息抽取服务，支持领域分类、意图识别和实体抽取",
    version="1.0.0"
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [{"role": "user", "content": user_prompt}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": response_model.model_json_schema(),
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            if (response.choices[0].message.tool_calls and
                    len(response.choices[0].message.tool_calls) > 0):
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return response_model.model_validate_json(arguments)
            else:
                return None
        except Exception as e:
            print(f"API调用错误: {e}")
            return None


class IntentDomainNerTask(BaseModel):
    """信息抽取任务定义"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather',
    'match', 'map', 'website', 'news', 'message', 'contacts', 'translation',
    'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone',
    'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'] = Field(description="领域类别")
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL',
    'CLOSEPRICE_QUERY', 'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY',
    'DOWNLOAD', 'QUERY', 'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY',
    'SENDCONTACTS', 'DEFAULT', 'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'] = Field(description="意图类型")
    Src: Optional[str] = Field(None, description="出发地")
    Dest: Optional[str] = Field(None, description="目的地")


class ExtractionRequest(BaseModel):
    """API请求模型"""
    text: str = Field(..., description="需要抽取信息的文本内容")
    model_name: str = Field("THUDM/GLM-Z1-9B-0414", description="使用的大模型名称")


class ExtractionResponse(BaseModel):
    """API响应模型"""
    success: bool = Field(description="抽取是否成功")
    data: Optional[IntentDomainNerTask] = Field(None, description="抽取结果数据")
    error: Optional[str] = Field(None, description="错误信息")


@app.get("/")
async def root():
    """API根路径，返回服务信息"""
    return {
        "message": "信息抽取API服务正常运行",
        "version": "1.0.0",
        "available_models": ["THUDM/GLM-Z1-9B-0414"],
        "docs_url": "/docs"
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_info(request: ExtractionRequest):
    """
    信息抽取接口

    - **text**: 需要分析的文本内容
    - **model_name**: 使用的大模型名称（默认: THUDM/GLM-Z1-9B-0414）
    """
    try:
        agent = ExtractionAgent(model_name=request.model_name)
        result = agent.call(request.text, IntentDomainNerTask)

        if result:
            return ExtractionResponse(success=True, data=result)
        else:
            return ExtractionResponse(
                success=False,
                error="信息抽取失败，可能模型无法处理该类型文本"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "timestamp": "2025-10-17T00:00:00Z"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "promptapi:app",
        host="0.0.0.0",  # 允许外部访问
        port=8050,  # 服务端口
        reload=True  # 开发时启用热重载
    )
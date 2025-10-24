from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal
import openai
import json

# 初始化FastAPI应用
app = FastAPI(title="信息抽取服务", description="基于大语言模型的信息抽取API")

# 配置OpenAI客户端
client = openai.OpenAI(
    api_key="sk-4c44ef4112a04e65910dfdd56774f084",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 请求体模型
class TextRequest(BaseModel):
    text: str

# 响应模型定义
class IntentDomainNerResponse(BaseModel):
    domain: Literal['music', 'app', 'weather', 'bus']
    intent: Literal['OPEN', 'SEARCH', 'QUERY']
    Src: Optional[str] = None
    Des: List[str]


# ExtractionAgent类
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print('ERROR', response.choices[0].message)
            raise HTTPException(status_code=500, detail=f"信息抽取失败: {str(e)}")

# 创建全局agent实例
extraction_agent = ExtractionAgent(model_name="qwen-plus")

# 定义各个任务的Pydantic模型
class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY'] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: List[str] = Field(description="目的地")

@app.post("/extract/intent-domain-ner", response_model=IntentDomainNerResponse)
async def extract_intent_domain_ner(request: TextRequest):
    """
    领域识别+意图识别+实体识别
    """
    try:
        result = extraction_agent.call(request.text, IntentDomainNerTask)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信息抽取失败: {str(e)}")

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}

# 启动命令:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# python -m uvicorn main:app --host 127.0.0.1 --port 8888 --reload
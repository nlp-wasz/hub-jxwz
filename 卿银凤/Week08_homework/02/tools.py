import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

# 初始化OpenAI客户端（使用阿里云百炼平台）
client = openai.OpenAI(
    # 填写你的 https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-12ca7074269f49e1afaf221d496f727f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


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
                        "required": response_model.model_json_schema()['required'],
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
        except:
            print('ERROR', response.choices[0].message)
            return None


class IntentDomainAndSlot(BaseModel):
    """对文本抽取领域类别、意图类别、实体类别"""
    domain: Literal["music", "app", "news", "bus"] = Field(description="领域")
    intent: Literal["OPEN", "SEARCH", "ROUTE", "QUERY"] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: Optional[str] = Field(description="目的地")

result = ExtractionAgent(model_name = "qwen-plus").call('北京到成都的汽车时刻表', IntentDomainAndSlot)
print(result)

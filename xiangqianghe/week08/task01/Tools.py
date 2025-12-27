import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal


client = openai.OpenAI(
    api_key="sk-gcuimbebodlqdoldluiucssywbxlatnoiyfwokewytvayzgt",

    base_url="https://api.siliconflow.cn/v1"
)



completion = client.chat.completions.create(
    model="THUDM/GLM-Z1-9B-0414",
    messages=[
        {"role": "system", "content": "你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图识别、实体标签"},
    ],
)
# result = completion.choices
# print(result)



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

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match', 'map', 'website', 'news', 'message', 'contacts', 'translation', 'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY', 'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY', 'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT', 'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Dec: Optional[str] = Field(description="目的地")
result = ExtractionAgent(model_name = "THUDM/GLM-Z1-9B-0414").call("你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？", IntentDomainNerTask)
print(result)

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match', 'map', 'website', 'news', 'message', 'contacts', 'translation', 'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY', 'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY', 'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT', 'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'] = Field(description="意图")
    slots: Literal[str] = Field(description="出发地")
    Dec: Optional[str] = Field(description="目的地")
result = ExtractionAgent(model_name = "THUDM/GLM-Z1-9B-0414").call("嗯咯鸡爪怎么做的。", IntentDomainNerTask)
print(result)

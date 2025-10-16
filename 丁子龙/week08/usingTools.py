import openai
import json
import os
from pydantic import BaseModel, Field
from typing import List,Dict,Any
from typing_extensions import Literal

client = openai.OpenAI(
    api_key="sk-ea07bf0880504b75a31b1bce38437fcf", # https://bailian.console.aliyun.com/?tab=model#/api-key
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


INTENTS = [
    "OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY",
    "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY",
    "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT",
    "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"
]

DOMAINS = [
    "music", "app", "radio", "lottery", "stock", "novel", "weather", "match",
    "map", "website", "news", "message", "contacts", "translation", "tvchannel",
    "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train",
    "poetry", "flight", "epg", "health", "email", "bus", "story"
]

SLOT_NAMES = [
    "code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole",
    "location_country", "location_area", "author", "startLoc_city", "season",
    "dishNamet", "media", "datetime_date", "episode", "teleOperator",
    "questionWord", "receiver", "ingredient", "name", "startDate_time",
    "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty",
    "area", "location_poi", "relIssue", "Dest", "content", "keyword", "target",
    "startLoc_area", "tvchannel", "type", "song", "queryField", "awayName",
    "headNum", "homeName", "decade", "payment", "popularity", "tag",
    "startLoc_poi", "date", "startLoc_province", "endLoc_province",
    "location_city", "absIssue", "utensil", "scoreDescr", "dishName",
    "endLoc_area", "resolution", "yesterday", "timeDescr", "category",
    "subfocus", "theatre", "datetime_time"
]


class Tools(BaseModel):
    """进行意图识别、领域识别和实体识别"""
    intents: str = Field(description="用户意图，必须是以下之一: " + ", ".join(INTENTS))
    domains: str = Field(description="所属领域，必须是以下之一: " + ", ".join(DOMAINS))
    slots: Dict[str, Any] = Field(
        description="抽取的槽位（实体）键值对。键必须是以下之一: " + ", ".join(SLOT_NAMES) +
            "；值可以是字符串、数字、列表等，根据实际语义决定。",
        examples=[{
      "area": "大陆",
      "category": "动漫",
      "datetime_time": "现在"
    }]
    )
# result = ExtractionAgent(model_name = "qwen-plus").call("张绍刚的综艺节目", Tools)
# print(result)

def usingTools(text):
    result = ExtractionAgent(model_name = "qwen-plus").call(text, Tools)
    return result



# print(Tools.model_json_schema())




















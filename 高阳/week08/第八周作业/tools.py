import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from typing_extensions import Literal



# 初始化FastAPI应用
app = FastAPI()

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-5ebc25ad675b4a77b1c27549f485f51c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match', 'map', 'website', 'news', 'message', 'contacts', 'translation', 'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story']
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY', 'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY', 'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT', 'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION']
    slots: List[str] = []
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tools = self._generate_tools()

    def _generate_tools(self):
        # 根据IntentDomainNerTask模型生成工具定义
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "IntentDomainNerTask",
                    "description": "对文本抽取领域类别、意图类型、实体标签",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "description": "领域",
                                "title": "Domain",
                                "type": "string",
                                "enum": ["music", "app", "radio"]
                            },
                            "intent": {
                                "description": "意图",
                                "title": "Intent",
                                "type": "string",
                                "enum": ["OPEN", "SEARCH", "REPLAY_ALL"]
                            },
                            "slots": {
                                "description": "实体",
                                "title": "Slots",
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["domain", "intent"]
                    }
                }
            }
        ]
        return tools

    def call(self, user_prompt: str) -> Optional[IntentDomainNerTask]:
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=150
        )

        try:
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return IntentDomainNerTask(**json.loads(arguments))
            else:
                return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





@app.post("/extract", response_model=IntentDomainNerTask)
def extract(user_prompt: str):
    try:
        agent = ExtractionAgent(model_name="qwen-plus")
        result = agent.call(user_prompt)
        if result:
            return result
        else:
            raise HTTPException(status_code=400, detail="No valid result extracted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


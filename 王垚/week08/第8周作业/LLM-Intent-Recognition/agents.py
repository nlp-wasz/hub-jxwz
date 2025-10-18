import openai
from typing import Type
from pydantic import BaseModel
import json
from config import settings


class ExtractionAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.client = openai.OpenAI(
            api_key=settings.LLM_OPENAI_API_KEY,
            base_url=settings.LLM_OPENAI_SERVER_URL,
        )

    def extract_with_tool(self, user_prompt: str, response_model: Type[BaseModel]):
        """使用Function Calling进行信息抽取"""
        messages = [{"role": "user", "content": user_prompt}]

        tools = [{
            "type": "function",
            "function": {
                "name": response_model.model_json_schema()['title'],
                "description": response_model.model_json_schema()['description'],
                "parameters": {
                    "type": "object",
                    "properties": response_model.model_json_schema()['properties'],
                },
            }
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f'Tool调用错误: {e}')
            return None

    def extract_with_prompt(self, text: str):
        """使用提示词工程进行信息抽取"""
        system_prompt = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签..."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        try:
            content = response.choices[0].message.content
            # 从返回内容中提取JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return IntentDomainNerTask(**data)
            return None
        except Exception as e:
            print(f'Prompt解析错误: {e}')
            return None

    def call(self, text: str, method: str = "tool"):
        """统一调用接口"""
        if method == "tool":
            return self.extract_with_tool(text, IntentDomainNerTask)
        else:
            return self.extract_with_prompt(text)
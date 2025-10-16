import random
from pydantic import BaseModel, Field
import openai
import json
from typing import List, Dict, Optional, Union
import pdb

class SlotItem(BaseModel):
    """实体标签项"""
    label: str = Field(description="标签名称")
    entity: str = Field(description="实体内容")


class ExtractionAgent:

    def __init__(self, api_key, base_url, model_name, prompt_str: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None

    def get_model_client(self) -> openai.OpenAI:
        if (self.client == None):
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def call(self, user_prompt, response_model):
        system_prompt = self.get_pormpt_str()
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema().get('title', ''),
                    "description": response_model.model_json_schema().get('description', ''),
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema().get('properties', ''),
                        "required": response_model.model_json_schema().get('required', []),
                    },
                }
            }
        ]
        client = self.get_model_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            result = response_model.model_validate_json(arguments)
            # 转换为期望的字典格式
            return result.model_dump_json()
        except:
            print('ERROR', response.choices[0].message)
            return None

    def get_pormpt_str(self) -> str:
        intents = []
        slots = []
        with open("../Week07/02-joint-bert-training-only/data/intents.txt", 'r', encoding='utf-8') as f:
            intents = [ins.strip() for ins in f if ins.strip()]

        with open("../Week07/02-joint-bert-training-only/data/slots.txt", 'r', encoding='utf-8') as f:
            slots = [ins.strip() for ins in f if ins.strip()]

        intent_str = "\n".join([f"- {intent}" for intent in intents])
        slot_str = "\n".join([f"- {slot}" for slot in slots])

        return f"""
        你是一个自然语言理解专家，请分析用户的输入，识别用户意图，并提取相关实体标签信息，
        可使用意图选项为:
        {intent_str}
        标签可选项为：
        {slot_str}
        请按照以下JSON格式输出,不要输出其他多余的信息：
        {{
            "text":"用户输入信息",
            "intent": "意图类别",
            "slots": [
            {{"label": "标签名1", "entity": "实体值1"}},
            {{"label": "标签名2", "entity": "实体值2"}}
            ]
        }}
                """.strip()


class Text(BaseModel):
    """文本问答内容解析"""
    text: str = Field(description="输入信息")
    intent: str = Field(description="意图识别")
    slots: List[Union[SlotItem, str]] = Field(description="实体标签，可以是SlotItem对象或字符串")


if __name__ == '__main__':
    exagent = ExtractionAgent(
        api_key="sk-e35a94e9130a4a71b9a9c99389275eaa",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus"
    )
    sentence = []

    with open("../Week07/02-joint-bert-training-only/data/sentences.txt", 'r', encoding='utf-8') as f:
        sentence = [ins.strip() for ins in f if ins.strip()]

    for i in range(0, 10):
        index = random.randint(0, len(sentence) - 1)
        print(exagent.call(sentence[index], Text))

# 其他方法
# https://www.promptingguide.ai/techniques

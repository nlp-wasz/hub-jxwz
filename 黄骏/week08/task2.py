import os
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

client = openai.AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

class ExtractionAgent:
    def __init__(self, model):
        self.model = model

    async def call(self, user_prompt, response_model):
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
                    "name": response_model.model_json_schema()["title"],
                    "description": response_model.model_json_schema()["description"],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()["properties"],
                    }
                }
            }
        ]

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print("ERROR: ", response.choices[0].message)
            return None

def read_txt_to_tuple(file_path: str) -> tuple:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            items = {line.strip() for line in f if line.strip()}
        return tuple(sorted(items))
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {file_path}")
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {str(e)}")

# 读取领域、意图和实体
domains = read_txt_to_tuple("domains.txt")
intents = read_txt_to_tuple("intents.txt")
slots = read_txt_to_tuple("slots.txt")

class NerTask(BaseModel):
    """实现领域识别 + 意图识别 + 实体识别"""
    domain: Literal[*domains] = Field(description="领域")
    intent: Literal[*intents] = Field(description="意图")
    slot: Literal[*slots] = Field(description="实体名")


async def extract_ner(user_prompt: str) -> Optional[NerTask]:
    """
    提取用户输入的领域、意图和实体信息

    参数:
        user_prompt: 用户输入的文本

    返回:
        包含domain、intent、slot的NerTask实例，失败时返回None
    """
    agent = ExtractionAgent(model="qwen-plus-latest")
    return await agent.call(user_prompt, NerTask)
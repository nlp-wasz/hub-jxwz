import json
from typing import Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
load_dotenv()

client = OpenAI(
    api_key=os.getenv('ALIBABA_BAILIAN_API_KEY'),
    base_url=os.getenv('ALIBABA_BAILIAN_API_BASE')
)

class EnhancedExtractionAgent:
    def __init__(self, system_prompt,model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def call_with_examples(self, user_prompt: str, response_model: BaseModel, examples: list = None):
        """
        结合少样本提示和工具调用的增强方法
        """
        # 构建消息序列
        messages = []
        # 1. 添加系统提示（可选）
        system_message = {
            "role": "system",
            "content": self.system_prompt
        }
        messages.append(system_message)

        # 2. 添加少样本示例（如果提供）
        if examples:
            for example in examples:
                if isinstance(example, HumanMessage):
                    messages.append({"role": "user", "content": example.content})
                elif isinstance(example, AIMessage) and hasattr(example, 'tool_calls'):
                    # 转换AIMessage为API格式
                    tool_calls = []
                    for tool_call in example.tool_calls:
                        tool_calls.append({
                            "id": tool_call.get('id', 'call_1'),
                            "type": "function",
                            "function": {
                                "name": tool_call.get('name', response_model.model_json_schema()['title']),
                                "arguments": json.dumps(tool_call.get('args', {}))
                            }
                        })
                    messages.append({
                        "role": "assistant",
                        "content": example.content,
                        "tool_calls": tool_calls
                    })
                elif isinstance(example, ToolMessage):
                    messages.append({
                        "role": "tool",
                        "content": example.content,
                        "tool_call_id": example.tool_call_id
                    })

        # 3. 添加当前用户查询
        messages.append({"role": "user", "content": user_prompt})

        # 4. 构建工具定义
        # model_json_schema用于获取 Pydantic 模型的 JSON Schema 表示
        tools = [{
            "type": "function",
            "function": {
                "name": response_model.model_json_schema()['title'],
                "description": response_model.model_json_schema()['description'],
                "parameters": {
                    "type": "object",
                    "properties": response_model.model_json_schema()['properties'],
                    "required": response_model.model_json_schema()['required'],
                }
            }
        }]

        try:
            # 5. 调用API（强制使用工具）
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": response_model.model_json_schema()['title']}},
            )

            # 6. 解析响应
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return response_model.model_validate_json(arguments)
            else:
                # 回退到内容解析
                content = response.choices[0].message.content
                if "{" in content and "}" in content:
                    json_str = content[content.find("{"):content.rfind("}") + 1]
                    return response_model.model_validate_json(json_str)
                raise ValueError("未返回结构化数据")

        except Exception as e:
            print(f'错误: {e}')
            return None
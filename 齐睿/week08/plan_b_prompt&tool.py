import json
from typing import Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Dict, Any
load_dotenv()

client = OpenAI(
    api_key=os.getenv('ALIBABA_BAILIAN_API_KEY'),
    base_url=os.getenv('ALIBABA_BAILIAN_API_BASE')
)


class EnhancedExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call_with_examples(self, user_prompt: str, response_model: BaseModel, examples: list = None):
        """
        结合少样本提示和工具调用的增强方法
        """
        # 构建消息序列
        messages = []
        system_prompt = """
        你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
        - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
        - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
        - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

        最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。
        ```json
        {{
            "domain": ,
            "intent": ,
            "slots": {{
              "实体类别": "实体名词",
            }}
        }}
        ```
        """
        # 1. 添加系统提示（可选）
        system_message = {
            "role": "system",
            "content": system_prompt
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


# 使用示例
def main():
    agent = EnhancedExtractionAgent(model_name="qwen-plus")

    # 定义少样本示例
    few_shot_examples = [
        HumanMessage("鱼香肉丝，怎么炒？"),
        AIMessage(
            "",
            tool_calls=[{
                "name": "IntentClass",
                "args": {
                    "text": "鱼香肉丝，怎么炒？",
                    "domain": "cookbook",
                    "intent": "QUERY",
                    "slots": {
                        "dishName": "鱼香肉丝"
                    }
                },
                "id": "1"
            }]
        ),
        ToolMessage("", tool_call_id="1"),
        HumanMessage("骨折了怎么办？"),
        AIMessage(
            "",
            tool_calls=[{
                "name": "IntentClass",
                "args": {
                    "text": "骨折了怎么办？",
                    "domain": "health",
                    "intent": "QUERY",
                    "slots": {
                        "keyword": "骨折"
                    }
                },
                "id": "2"
            }]
        ),
        ToolMessage("", tool_call_id="2")
    ]

    # Pydantic模型
    class IntentClass(BaseModel):
        """意图识别"""

        text: str = Field(description="用户输入的原始文本，即待识别意图的文本。", default="")
        domain: str = Field(description="领域类别，按照给出的类别进行识别和划分，给出所属的领域。")
        intent: str = Field(description="意图类别，按照给出的类别进行识别和划分，给出所属的意图。")
        slots: Dict[str, Any] = Field(
            description="实体插槽，将原始文本中的实体提取出来，一般以\"name\": \"XXXX\"形式存放,XXXX代表原始实体文本")  # 动态槽位参数

        class Config:
            # 允许额外字段但会触发验证警告
            extra = "allow"
    # 测试调用
    print("=== 结合少样本提示的测试 ===")

    # 测试1：使用少样本示例
    result1 = agent.call_with_examples(
        "包皮过长，怎么办？哦。",
        IntentClass,
        examples=few_shot_examples
    )
    print("带示例的结果:", result1)

    print("\n" + "=" * 50 + "\n")

    # 测试2：不使用示例（纯工具调用）
    result2 = agent.call_with_examples(
        "包皮过长，怎么办？哦。",
        IntentClass,
        examples=None  # 不使用示例
    )
    print("纯工具调用的结果:", result2)


if __name__ == "__main__":
    main()
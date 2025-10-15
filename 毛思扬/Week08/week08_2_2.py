from pydantic import BaseModel
from typing import List
import openai
import json

# 使用讲解大模型开发流程（提示词、tools、coze/dify）尝试写一下解决意图识别 + 领域识别 + 实体识别的过程。最终效果替代02-joint-bert-training-only
# 可以优先使用coze，不部署dify。
client = openai.OpenAI(
    api_key="sk-0a1a2ce5cc494c95b9b1bc3ccb3fc5ba",
    base_url="https://api.deepseek.com",
)


class AnalysisResult(BaseModel):
    """用户输入分析结果模型"""
    intent: List[str]
    domain: List[str]
    entities: List[str]


tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_user_input",
            "description": "分析用户输入，进行意图识别、领域识别和实体识别",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "识别到的用户意图，如：查询信息、执行操作、获取帮助、投诉建议等"
                    },
                    "domain": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "识别到的领域，如：生活服务、金融服务、医疗健康、教育学习、娱乐休闲、技术支持等"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "提取到的实体列表"
                    }
                },
                "required": ["intent", "domain", "entities"]
            }
        }
    }
]


def analyze_user_input_tools(input_text: str) -> AnalysisResult:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """
            你是一个智能助手，需要对用户的输入进行三层分析：

            1. 意图识别（Intent Recognition）
            2. 领域识别（Domain Recognition）  
            3. 实体识别（Entity Recognition）

            请使用 analyze_user_input 工具来返回分析结果。
            """},
            {"role": "user",
             "content": input_text},
        ],
        tools=tools,
        tool_choice="auto",
        stream=False
    )
    # 检查并处理工具调用结果
    tool_call = response.choices[0].message.tool_calls[0]
    # 使用pydantic验证和解析结果
    result_data = json.loads(tool_call.function.arguments)
    return AnalysisResult(**result_data)

# res = analyze_user_input_tools("""在自然语言处理领域，大型语言模型（LLM）如 GPT-3、BERT 等已经取得了显著的进展，它们
# 能够生成连贯、自然的文本，回答问题，并执行其他复杂的语言任务。然而，这些模型存在一
# 些固有的局限性，如“模型幻觉问题”、“时效性问题”和“数据安全问题”。为了克服这些限制
# ，检索增强生成（RAG）技术应运而生。""")
#
# print( res)




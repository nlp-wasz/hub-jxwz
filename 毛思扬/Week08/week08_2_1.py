import json
from typing import List

import openai
from pydantic import BaseModel

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


def analyze_user_input(input_text: str) -> AnalysisResult:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """"
            你是一个智能助手，需要对用户的输入进行三层分析：

    1. 意图识别（Intent Recognition）
    识别用户想要执行的操作，包括但不限于：
    - 查询信息（如：查询天气、查询订单状态）
    - 执行操作（如：预订酒店、发送邮件）
    - 获取帮助（如：询问功能、请求指导）
    - 投诉建议（如：反馈问题、提出建议）

    2. 领域识别（Domain Recognition）
    识别对话所属的专业领域，包括但不限于：
    - 生活服务（餐饮、住宿、出行、购物等）
    - 金融服务（银行、保险、投资理财等）
    - 医疗健康（病症咨询、预约挂号等）
    - 教育学习（课程咨询、学习资料等）
    - 娱乐休闲（电影、音乐、游戏等）
    - 技术支持（软件使用、设备故障等）

    3. 实体识别（Entity Recognition）
    从用户输入中提取关键信息实体，包括但不限于：
    - 时间（如：明天、2023年10月1日、下午3点）
    - 地点（如：北京、上海浦东机场、家里）
    - 人物（如：张三、李医生、客服小王）
    - 组织机构（如：阿里巴巴、工商银行、协和医院）
    - 产品服务（如：iPhone14、支付宝、体检套餐）
    - 数量金额（如：100元、3份、2小时）

       请严格按照以下JSON格式输出分析结果，不要包含其他文字：
            {
              "intent": ["识别到的意图列表"],
              "domain": ["识别到的领域列表"], 
              "entities": ["提取到的实体列表"]
            }
    用户输入：
            """},
            {"role": "user", "content": input_text},
        ],
        stream=False
    )

    # 获取模型响应
    response_text = response.choices[0].message.content
    # 解析为 AnalysisResult 对象
    analysis_result = AnalysisResult(**json.loads(response_text))
    return analysis_result

# res=analyze_user_input("""在自然语言处理领域，大型语言模型（LLM）如 GPT-3、BERT 等已经取得了显著的进展，它们
# 能够生成连贯、自然的文本，回答问题，并执行其他复杂的语言任务。然而，这些模型存在一
# 些固有的局限性，如“模型幻觉问题”、“时效性问题”和“数据安全问题”。为了克服这些限
# 制，检索增强生成（RAG）技术应运而生。""")
# print(res)
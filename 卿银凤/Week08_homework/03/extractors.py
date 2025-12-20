import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import Literal

# 初始化OpenAI客户端（使用阿里云百炼平台）
client = openai.OpenAI(
    api_key="sk-12ca7074269f49e1afaf221d496f727f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class IntentDomainAndSlot(BaseModel):
    """对文本抽取领域类别、意图类别、实体类别"""
    domain: Literal["music", "app", "news", "bus"] = Field(description="领域")
    intent: Literal["OPEN", "SEARCH", "ROUTE", "QUERY"] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: Optional[str] = Field(description="目的地")


class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.client = client

    def extract_with_prompt(self, text: str) -> Dict[str, Any]:
        """使用提示词方法进行信息抽取"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": """你是一个专业的信息抽取专家，请对下面的文本抽取他的领域类别、意图类别、实体类别
                - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
                - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
                - 待选的实体类别：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
                最终输出格式填充下面的json
                ```json  domain是领域标签， intent是意图标签， slots是实体识别的结果和标签
                {
                    "domain": ,
                    "intent": ,
                    "slots": {
                        "待选实体": "实体名词"
                    }
                }
                ```
                """},
                {"role": "user", "content": text},
            ],
        )
        print(completion.choices[0].message.content)
        result_content = completion.choices[0].message.content
        print(result_content)
        try:
            # 尝试解析JSON响应
            result = json.loads(result_content)
            return result
        except json.JSONDecodeError:
            # 如果返回的不是标准JSON，返回原始内容
            return {
                "domain": "unknown",
                "intent": "unknown",
                "slots": {},
                "raw_response": result_content
            }

    def extract_with_tools(self, text: str) -> Optional[IntentDomainAndSlot]:
        """使用工具方法进行信息抽取"""
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": IntentDomainAndSlot.model_json_schema()['title'],
                    "description": IntentDomainAndSlot.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": IntentDomainAndSlot.model_json_schema()['properties'],
                        "required": IntentDomainAndSlot.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return IntentDomainAndSlot.model_validate_json(arguments)
        except Exception as e:
            print(f'ERROR: {e}')
            print('Response:', response.choices[0].message)
            return None


# 创建全局提取器实例
extraction_agent = ExtractionAgent(model_name="qwen-plus")
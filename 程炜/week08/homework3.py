from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json

app = FastAPI()

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key="sk-0c73bfbd65c64592bbdc60362576aa90",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TextRequest1(BaseModel):
    text: str  # 用户输入的待分析文本


@app.post("/extract/llm")
async def extract_info1(request: TextRequest1):
    """
    信息抽取接口
    - 输入: 任意文本
    - 输出: 结构化抽取结果(领域/意图/实体)
    """
    # 系统提示词（同原始代码）
    system_prompt = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。"""  # 这里放入完整的系统提示

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text},
            ],
        )

        # 解析返回的JSON结果
        result = json.loads(completion.choices[0].message.content)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

class ExtractionAgent:
    """智能体封装"""

    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def extract(self, text: str, response_model: BaseModel):
        """通用抽取方法"""
        messages = [{"role": "user", "content": text}]

        tools = [{
            "type": "function",
            "function": {
                "name": response_model.model_json_schema()['title'],
                "description": response_model.model_json_schema()['description'],
                "parameters": response_model.model_json_schema(),
            }
        }]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f"Error: {e}")
            return None


# 定义数据模型
class IntentDomainNerTask(BaseModel):
    """文本信息抽取结果"""
    domain: Literal['music', 'app', 'weather', 'bus', 'map'] = Field(description="领域类别")
    intent: Literal['OPEN', 'SEARCH', 'QUERY', 'ROUTE'] = Field(description="意图类型")
    Src: Optional[str] = Field(description="出发地")
    Dest: List[str] = Field(description="目的地列表")


class TextRequest2(BaseModel):
    text: str = Field(..., example="从科大讯飞到天鹅湖怎么走能不走高速", description="待分析文本")


@app.post("/extract/agent", response_model=IntentDomainNerTask)
async def extract_info2(request: TextRequest2):
    """
    信息抽取接口
    - 输入: 任意文本
    - 输出: 结构化抽取结果(领域/意图/实体)
    """
    agent = ExtractionAgent()
    result = agent.extract(request.text, IntentDomainNerTask)

    if not result:
        raise HTTPException(status_code=400, detail="信息抽取失败")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
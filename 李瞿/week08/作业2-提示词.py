from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import openai
import json

# 初始化FastAPI应用
app = FastAPI(title="信息抽取服务", description="基于大语言模型的信息抽取API")

# 配置OpenAI客户端
client = openai.OpenAI(
    api_key="sk-4c44ef4112a04e65910dfdd56774f084",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 请求体模型
class TextRequest(BaseModel):
    text: str

# 响应体模型
class ExtractionResponse(BaseModel):
    domain: str
    intent: str
    slots: Dict[str, str]

# 系统提示词模板
SYSTEM_PROMPT = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

json { "domain": "领域标签", "intent": "意图标签", "slots": { "实体标签": "实体名词" } }
"""

@app.post("/extract", response_model=ExtractionResponse)
async def extract_info(request: TextRequest):
    """
    信息抽取接口

    Args:
        request: 包含待处理文本的请求体

    Returns:
        ExtractionResponse: 抽取的领域、意图和实体信息
    """
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.text},
            ],
            temperature=0.1
        )

        # 获取模型响应
        response_content = completion.choices[0].message.content

        # 提取JSON部分
        start_idx = response_content.find("json")
        end_idx = response_content.rfind("")

        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx+7:end_idx].strip()
            result = json.loads(json_str)
            return ExtractionResponse(**result)
        else:
            # 如果没有找到代码块，尝试直接解析整个响应
            result = json.loads(response_content)
            return ExtractionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信息抽取失败: {str(e)}")

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}

# 启动命令示例:
# uvicorn main:app --host 0.0.0.0 --port 8888 --reload
# python -m uvicorn main:app --host 127.0.0.1 --port 8888 --reload

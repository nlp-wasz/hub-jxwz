# promot_backend.py
import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# 数据模型
class AnalyzeRequest(BaseModel):
    sentence: str
    model: Optional[str] = "glm-4"
    temperature: Optional[float] = 0.1

class AnalyzeResponse(BaseModel):
    sentence: str
    domain: str
    intent: str
    slots: Dict[str, Any]

class BatchAnalyzeRequest(BaseModel):
    sentences: List[str]
    model: Optional[str] = "glm-4"
    temperature: Optional[float] = 0.1

class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]

# 智谱AI客户端
class ZhipuClient:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
    
    def chat(self, message, model="glm-4", temperature=0.1):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"错误: {e}"

# 解析JSON响应
def parse_json_response(response_text: str) -> Dict[str, Any]:
    print(f"调试 - 原始响应: {response_text}")
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        return json.loads(json_str)
    except Exception:
        return {"domain": "unknown", "intent": "unknown", "slots": {}}

# 创建FastAPI应用
app = FastAPI(title="句子分析API - Prompt方式")

# 添加CORS中间件，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端
zhipu_client = None

@app.on_event("startup")
async def startup_event():
    global zhipu_client
    api_key = os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        raise ValueError("请设置ZHIPUAI_API_KEY环境变量")
    zhipu_client = ZhipuClient(api_key)

# API接口
@app.get("/")
async def root():
    if zhipu_client is not None:
        return {"message": "句子分析API服务已启动", "status": "running"}
    else:
        return {"message": "句子分析API服务未启动", "status": "error"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    if zhipu_client is not None:
        return {
            "status": "healthy",
            "client_initialized": True
        }
    else:
        return {
            "status": "unhealthy", 
            "client_initialized": False,
            "error": "ZhipuAI客户端未初始化"
        }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentence(request: AnalyzeRequest):
    """分析单个句子 - 主要接口"""
    messages=[
            {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取它的领域类别、意图类型、实体标签
            - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
            - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / BUY/ SELL/ DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
            - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

            最终输出格式填充下面的json,domain 是领域标签， intent 是意图标签,slots 是实体识别结果和标签。

            ```json
            {
                "domain": ,
                "intent": ,
                "slots": {
                "待选实体": "实体名词",
                }
            }
            ```
            """},
                    {"role": "user", "content": request.sentence},
                ]
    
    response_text = zhipu_client.chat(
        message=messages, 
        model=request.model, 
        temperature=request.temperature
    )
    
    parsed_response = parse_json_response(response_text)
    
    return AnalyzeResponse(
        sentence=request.sentence,
        domain=parsed_response.get("domain", "unknown"),
        intent=parsed_response.get("intent", "unknown"),
        slots=parsed_response.get("slots", {})
    )

@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """批量分析句子"""
    results = []
    for sentence in request.sentences:
        single_request = AnalyzeRequest(sentence=sentence)
        result = await analyze_sentence(single_request)
        results.append(result)
    return BatchAnalyzeResponse(results=results)

# 启动后端服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
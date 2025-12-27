import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from typing_extensions import Literal

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

# 创建抽取智能体
class ExtractionAgent:
    def __init__(self, api_key, model_name: str):
        self.client=OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            )
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        # 增强提示词
        enhanced_prompt = f"""
        请仔细分析以下文本，识别领域、意图和所有相关的实体信息。
        特别注意提取：
        - 应用名称（如：微信、支付宝、携程、网易云音乐、等）
        - 歌曲名称、歌手名称
        - 城市名称、地点信息
        - 出发地、目的地
        - 菜名(如:爆椒牛柳、烩菜心、回锅肉等）实体
        - 时间信息（如：三分钟后、明天、下午三点等）
        - 实体若有多个则都提取出来
        文本：{user_prompt}
        """

        messages = [
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "parameters": response_model.model_json_schema()
                    },
                }
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            print(f"调试 - 完整返回: {response}")
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                print(f"调试 - 原始返回: {arguments}")
                
                # 手动处理 JSON 数据
                data = json.loads(arguments)
                
                # 确保所有字段都存在
                for field_name in response_model.model_fields:
                    if field_name not in data:
                        field_info = response_model.model_fields[field_name]
                        # 检查是否为列表类型
                        if (hasattr(field_info.annotation, '__origin__') and 
                            field_info.annotation.__origin__ is list):
                            data[field_name] = []
                        else:
                            data[field_name] = None
                    # 确保列表字段不为 None
                    elif data[field_name] is None:
                        field_info = response_model.model_fields[field_name]
                        if (hasattr(field_info.annotation, '__origin__') and 
                            field_info.annotation.__origin__ is list):
                            data[field_name] = []
                
                return response_model.model_validate(data)
            else:
                print('没有找到工具调用')
                return None    
        except json.JSONDecodeError as e:
            print(f'JSON 解析错误: {e}')
            print(f'原始参数: {arguments}')
            return None
        except Exception as e:
            print(f'其他错误: {e}')
            return None


class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus','train', 'navigation', 'cooking', 'other'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY', 'PLAY', 'NAVIGATION','ORDER','OTHER'] = Field(description="意图")
    #实体字段
    Src: Optional[str] = Field(description="出发地")
    Des: Optional[str] = Field(description="目的地")
    App: Optional[str] = Field(description="应用名称列表,如:微信、支付宝、携程、网易云音乐、B站等")
    Song: Optional[str] = Field(description="歌曲名称列表")
    Artist: Optional[str] = Field(description="歌手名称列表")
    City: Optional[str] = Field(description="城市名称列表")
    Location: Optional[str] = Field(description="地点信息")
    DishName: Optional[str] = Field(description="菜名列表")
    Datetime: Optional[str] = Field(description="时间信息，如:三分钟后、明天、早上、下午三点等")

# 创建FastAPI应用
app = FastAPI(title="句子分析API - Tools方式")

# 添加CORS中间件，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端
extraction_agent = None

@app.on_event("startup")
async def startup_event():
    global extraction_agent
    api_key = os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        print("❌ 错误: 请设置ZHIPUAI_API_KEY环境变量")
        return
    try:
        extraction_agent = ExtractionAgent(api_key, "glm-4")
        print("✅ ExtractionAgent客户端初始化成功")
    except Exception as e:
        print(f"❌ ExtractionAgent客户端初始化失败: {e}")

# API接口
@app.get("/")
async def root():
    if extraction_agent is not None:
        return {
            "message": "句子分析API服务已启动 (Tools方式)", 
            "status": "running",
            "client_initialized": True
        }
    else:
        return {
            "message": "句子分析API服务未正确初始化", 
            "status": "error",
            "client_initialized": False
        }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    if extraction_agent is not None:
        return {
            "status": "healthy",
            "client_initialized": True,
            "method": "tools"
        }
    else:
        return {
            "status": "unhealthy", 
            "client_initialized": False,
            "error": "ExtractionAgent客户端未初始化"
        }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentence(request: AnalyzeRequest):
    """分析单个句子 - Tools方式"""
    if extraction_agent is None:
        return AnalyzeResponse(
            sentence=request.sentence,
            domain="error",
            intent="error", 
            slots={"error": "ExtractionAgent客户端未初始化"}
        )
    
    # 使用Tools方式调用大模型
    result = extraction_agent.call(request.sentence, IntentDomainNerTask)
    
    if result is None:
        return AnalyzeResponse(
            sentence=request.sentence,
            domain="unknown",
            intent="unknown",
            slots={"error": "分析失败"}
        )
    
    # 将结果转换为slots格式
    slots = {}
    for field_name, field_value in result.model_dump().items():
        if field_name not in ['domain', 'intent'] and field_value is not None:
            if isinstance(field_value, list) and field_value:
                slots[field_name] = field_value
            elif not isinstance(field_value, list):
                slots[field_name] = field_value
    
    return AnalyzeResponse(
        sentence=request.sentence,
        domain=result.domain,
        intent=result.intent,
        slots=slots
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
    uvicorn.run(app, host="0.0.0.0", port=8008)  # 使用不同端口避免冲突

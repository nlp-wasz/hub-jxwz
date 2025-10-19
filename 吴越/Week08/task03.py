import time
import traceback
import uvicorn
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional,Literal
import openai
from fastapi import FastAPI,HTTPException
from task02 import ExtractionPrompt,ExtractionTools
import re
import json

app = FastAPI(
    title="信息抽取智能体服务",
    description="提供基于函数调用和Prompt工程的信息抽取",
    version="1.0.0"
)
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-fad1550b59d547ee83006bde2452e7bc",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionTools:

    def __init__(self,model_name:str):
        self.model_name = model_name

        self.domains = self.extract_categories("./domains.txt")
        self.intents = self.extract_categories("./intents.txt")


        class IntentDomainNerTask(BaseModel):
            """对文本抽取领域类别、意图类型、实体标签"""
            domain: Literal[tuple(self.domains)] = Field(description="领域")
            intent: Literal[tuple(self.intents)] = Field(description="意图")
            # 实体字段
            Src: Optional[str] = Field(default=None, description="来源")
            startDate_dateOrig: Optional[str] = Field (default=None, description="开始日期原始值")
            film: Optional[str] = Field(default=None, description="电影")
            endLoc_city: Optional[str] = Field(default=None, description="目的地城市")
            artistRole: Optional[str] = Field(default=None, description="艺术家角色")
            location_country: Optional[str] = Field(default=None, description="位置国家")
            location_area: Optional[str] = Field(default=None, description="位置区域")
            author: Optional[str] = Field(default=None, description="作者")
            startLoc_city: Optional[str] = Field(default=None, description="出发地城市")
            season: Optional[str] = Field(default=None, description="季节")
            dishNamet: Optional[str] = Field(default=None, description="菜品名称类型")
            media: Optional[str] = Field(default=None, description="媒体")
            datetime_date: Optional[datetime] = Field(default=None, description="日期时间-日期")
            episode: Optional[str] = Field(default=None, description="剧集")
            teleOperator: Optional[str] = Field(default=None, description="电信运营商")
            questionWord: Optional[str] = Field(default=None, description="疑问词")
            receiver: Optional[str] = Field(default=None, description="接收者")
            ingredient: Optional[str] = Field(default=None, description="食材")
            name: Optional[str] = Field(default=None, description="名称")
            startDate_time: Optional[datetime] = Field(default=None, description="开始时间")
            startDate_date: Optional[datetime] = Field(default=None, description="开始日期")
            location_province: Optional[str] = Field(default=None, description="位置省份")
            endLoc_poi: Optional[str] = Field(default=None, description="目的地兴趣点")
            artist: Optional[str] = Field(default=None, description="艺术家")
            dynasty: Optional[str] = Field(default=None, description="朝代")
            area: Optional[str] = Field(default=None, description="区域")
            location_poi: Optional[str] = Field(default=None, description="位置兴趣点")
            relIssue: Optional[str] = Field(default=None, description="相关问题")
            Dest: Optional[str] = Field(default=None, description="目的地")
            content: Optional[str] = Field(default=None, description="内容")
            keyword: Optional[str] = Field(default=None, description="关键词")
            target: Optional[str] = Field(default=None, description="目标")
            startLoc_area: Optional[str] = Field(default=None, description="出发地区域")
            tvchannel: Optional[str] = Field(default=None, description="电视频道")
            type: Optional[str] = Field(default=None, description="类型")
            song: Optional[str] = Field(default=None, description="歌曲")
            queryField: Optional[str] = Field(default=None, description="查询字段")
            awayName: Optional[str] = Field(default=None, description="客队名称")
            headNum: Optional[str] = Field(default=None, description="人数")
            homeName: Optional[str] = Field(default=None, description="主队名称")
            decade: Optional[str] = Field(default=None, description="年代")
            payment: Optional[str] = Field(default=None, description="支付方式")
            popularity: Optional[str] = Field(default=None, description="流行度")
            tag: Optional[str] = Field(default=None, description="标签")
            startLoc_poi: Optional[str] = Field(default=None, description="出发地兴趣点")
            date: Optional[str] = Field(default=None, description="日期")
            startLoc_province: Optional[str] = Field(default=None, description="出发地省份")
            endLoc_province: Optional[str] = Field(default=None, description="目的地省份")
            location_city: Optional[str] = Field(default=None, description="位置城市")
            absIssue: Optional[str] = Field(default=None, description="绝对问题")
            utensil: Optional[str] = Field(default=None, description="厨具")
            scoreDescr: Optional[str] = Field(default=None, description="分数描述")
            dishName: Optional[str] = Field(default=None, description="菜品名称")
            endLoc_area: Optional[str] = Field(default=None, description="目的地区域")
            resolution: Optional[str] = Field(default=None, description="分辨率")
            yesterday: Optional[str] = Field(default=None, description="昨天")
            timeDescr: Optional[str] = Field(default=None, description="时间描述")
            category: Optional[str] = Field(default=None, description="类别")
            subfocus: Optional[str] = Field(default=None, description="子焦点")
            theatre: Optional[str] = Field(default=None, description="剧院")
            datetime_time: Optional[datetime] = Field(default=None, description="日期时间-时间")

        self.IntentDomainNerTask = IntentDomainNerTask

    def extract_categories(self, filepath: str) -> List[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"文件地址{filepath}不存在")
            return []
    def chat(self,text):
        response_model = self.IntentDomainNerTask
        messages=[
            {
                "role":"user",
                "content":text,
            }
        ]

        tools=[
            {
                "type":"function",
                "function":{
                    "name":response_model.model_json_schema()['title'],
                    "description":response_model.model_json_schema().get('description',''),
                    "parameters":{
                        "type":response_model.model_json_schema()['type'],
                        "properties":response_model.model_json_schema().get('properties',{}),
                        "required": response_model.model_json_schema().get('required', []),
                    },
                }
            }
        ]

        response=client.chat.completions.create(
           model=self.model_name,
           messages=messages,
           tools=tools,
           tool_choice="auto",
           )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            result=response_model.model_validate_json(arguments)


            entities={}
            for field_name,field_value in result.dict().items():
                if field_name not in ['domain','intent'] and field_value is not None:
                    entities[field_name]=field_value

            return {
                "domain": result.domain,
                "intent": result.intent,
                "entities": entities,
                "success": True
            }

        except Exception as e:
            return {
                "domain": None,
                "intent": None,
                "entities": {},
                "success": False,
                "error": str(e)
            }

class ExtractionPrompt:
    """
    基于提示词工程的信息抽取，包括意图识别、领域识别和实体识别
    """
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.domains=self._extract_categories("./domains.txt")
        self.intents=self._extract_categories("./intents.txt")
        self.slots=self._extract_categories("./slots.txt")
        self.prompt=self._build_prompt()


    def _extract_categories(self,filepath:str) -> List[str]:
        try:
            with open(filepath,"r",encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"文件地址{filepath}不存在")
            return []

    def _build_prompt(self) -> str:
        domains="/".join(self.domains)
        intents="/".join(self.intents)
        slots="/".join(self.slots)

        return f"""
        你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
        - 待选的领域类别：{domains}
        - 待选的意图类别:{intents}
        - 待选的实体标签:{slots}
        最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。
        ```json
        {{
        "domain": ,
        "intent": ,
        "slots": {{
          "待选实体": "实体名词",}}
        }}
        ```
        """

    def chat(self,text:str) ->Optional[Dict[str, Any]]:

        try:
            completion=client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role":"system","content":self.prompt},
                    {"role":"user","content":text},
                ],
            temperature = 0.1,
            )
            result=completion.choices[0].message.content
            return result

        except Exception as e:
            print(f"信息抽取发生错误：{e}")
            return None

class Extract_Request(BaseModel):
    """
    请求格式
    """
    request_text: str= Field(..., description="需要抽取的文本")
    model_name: str = Field(default="qwen-plus", description="使用的模型名称")


class Extract_Response(BaseModel):
    """
    接口返回格式
    """
    domain: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None

@app.get("/")
async def root():
    return{"message":"信息抽取智能体服务已启动","status":"running"}

@app.post("/extract/funtion",response_model=Extract_Response)
async def extract_with_funtion(request: Extract_Request):
    """
    使用函数调用方式进行信息抽取
    """

    try:

        agent=ExtractionTools(request.model_name)
        result=agent.chat(request.request_text)
        return Extract_Response(**result)

    except Exception as err:
            raise HTTPException(status_code=500, detail=f"函数调用抽取失败: {str(err)}")




@app.post("/extract/Prompt",response_model=Extract_Response)
async def extract_with_Prompt(request: Extract_Request):
    """
    使用Prompt方式进行信息抽取
    """
    try:
        Prompt=ExtractionPrompt(request.model_name)
        result=Prompt.chat(request.request_text)
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return Extract_Response(**data)
        else:
            # 如果没有找到代码块，尝试直接解析整个响应
            try:
                data = json.loads(result)
                return Extract_Response(**data)
            except:
                return Extract_Response(
                    success=False,
                    error="无法解析模型响应"
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt工程抽取失败: {str(e)}")


@app.get("/health")
async def health():
    return{"status":"health","time":datetime.datetime.now()}

uvicorn.run(app, host="0.0.0.0", port=8002)


"""
$body = @{
    model_name = "qwen-plus"
    request_text = "帮我查询下从北京到天津的在2025年10月1日的火车票"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8002/extract/funtion" -Method POST -Body $body -ContentType "application/json"
"""
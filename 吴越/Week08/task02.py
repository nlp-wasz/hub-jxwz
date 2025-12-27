import openai
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from typing_extensions import Literal
from datetime import datetime


# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-fad1550b59d547ee83006bde2452e7bc",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#提示词
class ExtractionPrompt:
    """
    基于提示词工程的信息抽取，包括意图识别、领域识别和实体识别
    """
    def __init__(self):
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
                model="qwen-plus",
                messages=[
                    {"role":"system","content":self.prompt},
                    {"role":"user","content":text},
                ],
            temperature = 0.1,
            )
            return completion.choices[0].message.content

        except Exception as e:
            print(f"信息抽取发生错误：{e}")
            return None

#tools
class ExtractionTools:

    def __init__(self):

        self.domains = self.extract_categories("./domains.txt")
        self.intents = self.extract_categories("./intents.txt")


        class IntentDomainNerTask(BaseModel):
            """对文本抽取领域类别、意图类型、实体标签"""
            domain: Literal[tuple(self.domains)] = Field(description="领域")
            intent: Literal[tuple(self.intents)] = Field(description="意图")
            # 实体字段
            Src: Optional[str] = Field(default=None, description="来源")
            startDate_dateOrig: Optional[str] = Field(default=None, description="开始日期原始值")
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
                    },
                }
            }
        ]

        response=client.chat.completions.create(
           model="qwen-plus",
           messages=messages,
           tools=tools,
           tool_choice="auto",
           )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
                print(f"信息抽取发生错误：{e}")
                return None






#
result1=ExtractionPrompt().chat("糖醋鲤鱼怎么做啊？你只负责吃，c则c。")
print(result1)
#
# result2= ExtractionTools().chat("帮我查询下从北京到天津到武汉的汽车票")
# print(result2)








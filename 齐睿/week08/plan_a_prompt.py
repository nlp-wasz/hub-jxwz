"""
总结langchain结构化输出的主要问题：
    - 对于不支持结构化输出的模型支持效果欠佳，因为其输出是根据pydantic进行校验，但是其内部转化仍依靠用户输入的prompt中显式提示到与pydantic一致的结构。
    - 例如： 当提示词仅为：Tell me a joke about cats. Return the response in the JSON format.
    - 其最终会因为模型输出的json格式与pydantic不一致而导致错误，因为其只会返回一个joker字段的单字段Json
    - 实际上的提示词应为：


"""

import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()
# 获取环境变量
api_key = os.getenv('ALIBABA_BAILIAN_API_KEY')
base_url = os.getenv('ALIBABA_BAILIAN_API_BASE')
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url
model = init_chat_model("qwen-plus", model_provider="openai")


# messages = [
#     HumanMessage(content="你是谁？"),
# ]
#
# print(model.invoke(messages))

# 希望模型返回Pydantic对象，也就是格式化输出我需要的内容（通过Pydantic自动验证）
from pydantic import BaseModel, Field
from typing import Dict, Any

# Pydantic
class IntentClass(BaseModel):
    """意图识别"""

    text: str = Field(description="用户输入的原始文本，即待识别意图的文本。", default="")
    domain: str = Field(description="领域类别，按照给出的类别进行识别和划分，给出所属的领域。")
    intent: str = Field(description="意图类别，按照给出的类别进行识别和划分，给出所属的意图。")
    slots: Dict[str, Any] = Field(description="实体插槽，将原始文本中的实体提取出来，一般以\"name\": \"XXXX\"形式存放,XXXX代表原始实体文本") # 动态槽位参数
    class Config:
        # 允许额外字段但会触发验证警告
        extra = "allow"


# 使用更明确的提示词
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
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
])
structured_llm = model.with_structured_output(IntentClass)
chain = prompt | structured_llm
result = chain.invoke({"query": "鱼香肉丝，怎么炒？"})
print(json.dumps(result.dict(), indent=2, ensure_ascii=False))
# 其中的invoke相当于嵌套注入：structured_llm.invoke(prompt.invoke(input))，对于invoke，任何实现了Runnable接口的对象都可以调用 .invoke()

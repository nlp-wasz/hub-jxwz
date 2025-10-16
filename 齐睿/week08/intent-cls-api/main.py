from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import openai
from fastapi import FastAPI
from schema import IntentClass,IntentClassifyRequest, IntentClassifyResponse
from tool import EnhancedExtractionAgent
import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()
# 获取环境变量
api_key = os.getenv('ALIBABA_BAILIAN_API_KEY')
base_url = os.getenv('ALIBABA_BAILIAN_API_BASE')
dify_base_url = os.getenv('DIFY_BASE_URL')
dify_api_key = os.getenv('DIFY_INTENT_CLASSIFY')

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url
model = init_chat_model("qwen-plus", model_provider="openai")

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

app = FastAPI()
@app.post("/v1/intent-cls/zeroshot-prompt")
def zeroshot_prompt_classify(req: IntentClassifyRequest) -> IntentClassifyResponse:
    """
    利用零样本提示词进行意图分类

    :param req: 请求体
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    structured_llm = model.with_structured_output(IntentClass)
    chain = prompt | structured_llm
    result = chain.invoke({"query": req})
    print(json.dumps(result.dict(), indent=2, ensure_ascii=False))
    return IntentClassifyResponse(
        domain=result.domain,
        intent=result.intent,
        slots=result.slots
    )

@app.post("/v1/intent-cls/fewshot-prompt&tool")
def fewshot_prompt_and_tool(req: IntentClassifyRequest) -> IntentClassifyResponse:
    """
    利用少样本提示词结合强制工具调用进行意图分类
    :param req: 请求体
    """
    agent = EnhancedExtractionAgent(system_prompt, model_name="qwen-plus")
    # 定义少样本示例
    few_shot_examples = [
        HumanMessage("鱼香肉丝，怎么炒？"),
        AIMessage(
            "",
            tool_calls=[{
                "name": "IntentClass",
                "args": {
                    "text": "鱼香肉丝，怎么炒？",
                    "domain": "cookbook",
                    "intent": "QUERY",
                    "slots": {
                        "dishName": "鱼香肉丝"
                    }
                },
                "id": "1"
            }]
        ),
        ToolMessage("", tool_call_id="1"),
        HumanMessage("骨折了怎么办？"),
        AIMessage(
            "",
            tool_calls=[{
                "name": "IntentClass",
                "args": {
                    "text": "骨折了怎么办？",
                    "domain": "health",
                    "intent": "QUERY",
                    "slots": {
                        "keyword": "骨折"
                    }
                },
                "id": "2"
            }]
        ),
        ToolMessage("", tool_call_id="2")
    ]
    result = agent.call_with_examples(
        str(req),
        IntentClass,
        examples=few_shot_examples
    )
    print(json.dumps(result.dict(), indent=2, ensure_ascii=False))
    return IntentClassifyResponse(
        domain=result.domain,
        intent=result.intent,
        slots=result.slots
    )



@app.post("/v1/intent-cls/dify-api")
def dify_api_classify(req: IntentClassifyRequest) -> IntentClassifyResponse:
    """
    利用零样本提示词进行意图分类

    :param req: 请求体
    """
    result = intent_class(str(req))
    return IntentClassifyResponse(
        domain=result["domain"],
        intent=result["intent"],
        slots=result["slots"]
    )

def intent_class(query: str, user: str = "py-client", stream=False):
    # 1. 改路径
    url = f"{dify_base_url}/workflows/run"
    # 2. 加 X-USER-ID
    hdrs = {
        "Authorization": f"Bearer {dify_api_key}",
        "Content-Type": "application/json"
    }
    # 3. 改请求体
    payload = {
        "inputs": {"Query": query},  # 把用户问题塞进 inputs
        "response_mode": "streaming" if stream else "blocking",
        "user": user
    }

    r = requests.post(url, json=payload, headers=hdrs, stream=stream, timeout=600)

    if stream:
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                print(json.loads(line[5:])["data"]["outputs"]["answer"], end="")
    else:
        # 4. 取 outputs
        return r.json()["data"]["outputs"]["result"]

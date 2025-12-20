import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
print("API Key:", DASHSCOPE_API_KEY)

# 提示词 解决意图识别 + 领域识别 + 实体识别的过程


client = openai.OpenAI(
    api_key=DASHSCOPE_API_KEY,  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



def class_base_on_input(query_str):
    print("query_str", query_str)
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
            - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
            - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
            - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

            最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

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
            # {"role": "user", "content": "糖醋鲤鱼怎么做啊？你只负责吃，c则c。"},
            {"role": "user", "content": query_str},

        ],
    )
    print("\nZero-Shot Prompting")
    result=completion.choices[0].message.content
    print(result)
    return result


# class_base_on_input("糖醋鲤鱼怎么做啊？你只负责吃，c则c。")

# 其他方法
# https://www.promptingguide.ai/techniques
import openai
import json


client = openai.OpenAI(
    api_key="sk-gcuimbebodlqdoldluiucssywbxlatnoiyfwokewytvayzgt",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://api.siliconflow.cn/v1"
)


completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    messages=[
        {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图识别、实体标签
        - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
        - 待选的意图类型：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
        - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
       
       最终输出格式填充下面的json， domain是领域标签， intent是意图标签， slots是实体识别结果和标签
       ```json
       {
            "domain":,
            "intent":,
            "slots":{
                "待选实体"："实体名词"
            }
       }
       ```
       """},
        {"role":"user", "content": "从合肥到上海可以到哪坐车？"}
    ],
)
result = completion.choices
print(result)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    messages=[
        {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图识别、实体标签
        - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
        - 待选的意图类型：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
        - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

       最终输出格式填充下面的json， domain是领域标签， intent是意图标签， slots是实体识别结果和标签
       ```json
       {
            "domain":,
            "intent":,
            "slots":{
                "待选实体"："实体名词"
            }
       }
       ```
       """},
        {"role": "user", "content": "嗯咯鸡爪怎么做的。"}
    ],
)
result = completion.choices
print(result)










import openai
import json

client = openai.OpenAI(
    api_key="sk-d5501fae00944a8c852dba29be821720",
    base_url="https://api.deepseek.com/",
)

class QuestionAnalyzeAgent:
    def __init__(self):
        self.model_name = "deepseek-chat"

    def call(self, user_prompt):
        messages = [
            {
                "role": "system",
                "content": """
                    你是一个文本识别专家，需要从以下三方面识别给定的文本，返回识别结果：
                    1.领域识别；2.意图识别；3.实体识别
                    """
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "question_analyze", # 工具名字
                    "description": "识别用户提问的领域、意图、实体，分类到所提供的类别之一", # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story",
                            },
                            "intent": {
                                "type": "string",
                                "description": "待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION",
                            },
                            "entity": {
                                "type": "string",
                                "description": "待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time",
                            },
                        },
                        "required": ["domain", "intent", "entity"],  # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return json.loads(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

if __name__ == '__main__':
    result = QuestionAnalyzeAgent().call("帮我查询下从北京到天津到武汉的汽车票")
    print(result)


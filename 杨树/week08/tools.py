import openai

# 初始化 DeepSeek client
client = openai.OpenAI(
    api_key="sk-94609c22f9b949079f55863fa47d17cd",
    base_url="https://api.deepseek.com/"
)

# 定义工具函数（function calling）
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_user_input",
            "description": "将用户的输入语句进行领域识别,意图识别,实体识别",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "description": "- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story",
                        "type": "string",
                    },
                    "intent": {
                        "description": "- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION",
                        "type": "string",
                    },
                    "entity": {
                        "description": "- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time",
                        "type": "string",
                    },
                },
                "required": ["domain", "intent", "entity"],
            },
        },
    }
]

def analyze_user_input_tools(text: str):
    """
    对输入文本进行领域识别、意图识别、实体识别。
    返回包含 domain、intent、entity 的 JSON。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """
                你是一个语句识别助手，需要对用户的输入进行下面三种识别：
                1. 意图识别（Intent Recognition）
                2. 领域识别（Domain Recognition）  
                3. 实体识别（Entity Recognition）
                并返回分析结果。
                """},
                {"role": "user", "content": text},
            ],
            tools=tools,
            tool_choice="auto",
        )

        # 提取工具调用结果
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            arguments = message.tool_calls[0].function.arguments
            import json
            return json.loads(arguments)
        else:
            return {"error": "模型未返回工具调用结果", "raw": message}

    except Exception as e:
        return {"error": str(e)}


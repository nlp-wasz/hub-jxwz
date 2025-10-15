# task1_prompt.py
import os
from openai import OpenAI

# ZhipuAI Client Wrapper
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

# Sentence generator
def sentences_generator_expr(filename):
    """使用生成器表达式"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            return iter(lines)
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return iter([])

# Entry
if __name__ == "__main__":
    api_key = os.getenv('ZHIPUAI_API_KEY')
    zhipu_client = ZhipuClient(api_key)
    
    sentences_gen = sentences_generator_expr('sentences.txt')

    for sentence in sentences_gen:
        print(sentence)
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
                    {"role": "user", "content": sentence},
                ]
        response = zhipu_client.chat(message=messages)
        print(response)

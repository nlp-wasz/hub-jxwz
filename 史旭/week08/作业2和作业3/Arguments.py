import json
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from openai import OpenAI


class Arguments():
    # 定义 项目中所需要的参数信息
    def __init__(self):
        self.bert_model_dir = "../../models/google-bert/bert-base-chinese"
        self.bert_tokenizer_dir = "../../models/google-bert/bert-base-chinese"
        self.train_model_dir = "./training/train_model/"
        self.train_model_state_dir = "./training/state/state.pt"

        # 记录 用户提出的问题对应的  意图、领域、NER类型（类型与编码  之间的对应关系）
        self.domains = []
        self.domain_to_index = {}
        self.index_to_domain = {}
        with open("./data/domains.txt", "r", encoding="utf-8") as f:
            self.domains = [domain.strip() for domain in f.read().split("\n")]
            self.domain_to_index = {domain: index for index, domain in enumerate(self.domains)}
            self.index_to_domain = {index: domain for index, domain in enumerate(self.domains)}

        self.intents = []
        self.intent_to_index = {}
        self.index_to_intent = {}
        with open("./data/intents.txt", "r", encoding="utf-8") as f:
            self.intents = [intent.strip() for intent in f.read().split("\n")]
            self.intent_to_index = {intent: index for index, intent in enumerate(self.intents)}
            self.index_to_intent = {index: intent for index, intent in enumerate(self.intents)}

        self.slots = []
        self.slot_to_index = {}
        self.index_to_slot = {}

        self.slotsBIO = ["O"]
        self.slotBIO_to_index = {}
        self.index_to_slotBIO = {}
        with open("./data/slots.txt", "r", encoding="utf-8") as f:
            self.slots = [slot.strip() for slot in f.read().split("\n")]
            self.slot_to_index = {slot: index for index, slot in enumerate(self.slots)}
            self.index_to_slot = {index: slot for index, slot in enumerate(self.slots)}

            # 将 slots 中的所有类别 设置成 BIO类型
            for slot in self.slots:
                self.slotsBIO.extend(["B-" + slot, "I-" + slot])

            self.slotBIO_to_index = {slotBIO: index for index, slotBIO in enumerate(self.slotsBIO)}
            self.index_to_slotBIO = {index: slotBIO for index, slotBIO in enumerate(self.slotsBIO)}

        self.is_save_model = True
        self.max_seq_length = 32
        self.hidden_dim = 768
        self.cls_domain_dim = len(self.domain_to_index)
        self.cls_intent_dim = len(self.intent_to_index)
        self.ner_slot_dim = len(self.slotBIO_to_index)

        self.batch_size = 64  # 划分数据集批次 样本数量
        self.shuffle = True  # 批次样本是否打乱 获取

        self.lr = 2e-5  # 学习率
        self.device = None  # 设备
        self.epochs = 5  # 模型训练轮次

        self.hidden_dropout_prob = 0.1  # 随机丢弃神经元
        self.is_save_model = True
        self.is_save_model_state = True

        self.api_key = "sk-04ab3d7290e243dda1badc5a1d5ac858"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen-plus"

        # 提示词模板
        self.prompt = '''你是一个专业的意图识别与信息抽取专家。请对下面的输入文本进行领域、意图和实体识别，并完成三项任务：
        输入文本：
        {inputs}
        
        领域类别：[
            "music", "app", "radio", "lottery", "stock", "novel", "weather", "match",
            "map", "website", "news", "message", "contacts", "translation", "tvchannel",
            "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train",
            "poetry", "flight", "epg", "health", "email", "bus", "story"
        ]
        
        意图类别：[
            "OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY",
            "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY",
            "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT",
            "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"
        ]
    
        实体类别：[
            "code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole",
            "location_country", "location_area", "author", "startLoc_city", "season",
            "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord",
            "receiver", "ingredient", "name", "startDate_time", "startDate_date",
            "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi",
            "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel",
            "type", "song", "queryField", "awayName", "headNum", "homeName", "decade",
            "payment", "popularity", "tag", "startLoc_poi", "date", "startLoc_province",
            "endLoc_province", "location_city", "absIssue", "utensil", "scoreDescr",
            "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr", "category",
            "subfocus", "theatre", "datetime_time"
        ]
        
        1. 判断每条文本的【领域】（domain）
        2. 判断每条文本的【意图】（intent）
        3. 抽取每条文本中的【实体】（entities），包括：
           - 实体类型（必须使用 `entity_type` 字段）
           - 实体原文（必须使用 `text` 字段）
           - 起始位置 `start` 和结束位置 `end`（字符索引，从0开始）

        ### 输出格式要求（严格遵守）：
        - 必须返回一个 JSON 对象，包含字段 `"results"`，其值为列表
        - 列表中每个元素是一个对象，必须包含：
          - `"pre_text"`: 原始文本
          - `"domain"`: 领域
          - `"intent"`: 意图
          - `"entities"`: 实体列表，每个实体必须包含 `entity_type`, `text`, `start`, `end`
        - 实体字段名 **必须是 `entity_type` 而不是 `type`，必须是 `text` 而不是 `entity`**
        - 位置必须是整数，`start` ≤ `end`
        - 顺序必须与输入文本一致

        ### 示例输出结构（不要输出这个例子）：
        {{
          "results": [
            {{
              "pre_text": "你去玩手机了吧用英语怎么说",
              "domain": "translation",
              "intent": "TRANSLATION",
              "entities": [
                {{
                  "entity_type": "Src",
                  "text": "手机",
                  "start": 5,
                  "end": 7
                }}
              ]
            }}
          ]
        }}

        注意事项：
        - 必须处理每一个文本，不能遗漏
        - 如果某文本无实体，`entities` 返回空列表 []
        - 输出必须是合法 JSON，不要加额外说明
        - 直接输出 JSON，不要用代码块包裹
        '''

        # tools 工具
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "ClsAndNer",
                    "description": "批量识别多个文本的领域、意图和实体",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "description": "每条文本的识别结果列表，顺序与输入一致",
                                "title": "results",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pre_text": {
                                            "type": "string",
                                            "description": "用户输入的原始文本内容",
                                            "title": "pre_text"
                                        },
                                        "domain": {
                                            "type": "string",
                                            "description": "该文本所属的功能领域",
                                            "title": "domain"
                                        },
                                        "intent": {
                                            "type": "string",
                                            "description": "用户在该文本中表达的主要意图",
                                            "title": "intent"
                                        },
                                        "entities": {
                                            "type": "array",
                                            "description": "识别出的所有实体列表，每个实体包含类型、原文、起止位置",
                                            "title": "entities",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entity_type": {
                                                        "type": "string",
                                                        "description": "实体的类别标签"
                                                    },
                                                    "text": {
                                                        "type": "string",
                                                        "description": "实体在原始文本中的原始字符串"
                                                    },
                                                    "start": {
                                                        "type": "integer",
                                                        "description": "实体在原始文本中的起始字符索引（从0开始）"
                                                    },
                                                    "end": {
                                                        "type": "integer",
                                                        "description": "实体在原始文本中的结束字符索引（不包含）"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["pre_text", "domain", "intent", "entities"],
                    }
                }
            }
        ]

        # 结合 pydantic 的tools工具（自动解释参数）
        self.pydanticTools = [
            {
                "type": "function",
                "function": {
                    "name": BatchResult.model_json_schema()["title"],
                    "description": BatchResult.model_json_schema()["description"],
                    "parameters": {
                        "type": "object",
                        "properties": BatchResult.model_json_schema()["properties"],
                        "required": BatchResult.model_json_schema()["required"],
                    }
                }
            }
        ]


# 结合 pydantic 的tools工具模板对象（自动解释参数）
# 单个实体
class Entity(BaseModel):
    entity_type: Literal[
        "code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole",
        "location_country", "location_area", "author", "startLoc_city", "season",
        "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord",
        "receiver", "ingredient", "name", "startDate_time", "startDate_date",
        "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi",
        "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel",
        "type", "song", "queryField", "awayName", "headNum", "homeName", "decade",
        "payment", "popularity", "tag", "startLoc_poi", "date", "startLoc_province",
        "endLoc_province", "location_city", "absIssue", "utensil", "scoreDescr",
        "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr", "category",
        "subfocus", "theatre", "datetime_time"
    ] = Field(..., description="实体类别")
    text: str = Field(..., description="实体原文内容")
    start: int = Field(..., description="在原始文本中的起始位置")
    end: int = Field(..., description="在原始文本中的结束位置")


# 定义单条文本识别结果
class PydanticTools(BaseModel):
    '''对一条文本的领域、意图、实体识别结果'''
    pre_text: str = Field(..., description="用户输入的原始文本")

    domain: Literal[
        "music", "app", "radio", "lottery", "stock", "novel", "weather", "match",
        "map", "website", "news", "message", "contacts", "translation", "tvchannel",
        "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train",
        "poetry", "flight", "epg", "health", "email", "bus", "story"
    ] = Field(..., description="领域类别")

    intent: Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY",
    "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY",
    "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT",
    "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(..., description="意图类别")

    entities: List[Entity] = Field(description="识别出的所有实体")


# 定义批量结果容器（关键！）
class BatchResult(BaseModel):
    '''批量识别多个文本的领域、意图和实体'''
    results: List[PydanticTools] = Field(..., description="每条文本的识别结果列表，顺序与输入一致")


# 自定义Agent（自动完成 pydantic + tools + LLM 流程）
class AutoAgent:
    def __init__(self, args):
        self.args = args

    def call(self, pre_texts):
        # 加载 OpenAI
        llm = OpenAI(
            api_key=self.args.api_key,
            base_url=self.args.base_url
        )

        res = llm.chat.completions.create(
            model=self.args.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的assistant"},
                {"role": "user", "content": self.args.prompt.format(inputs=pre_texts)}
            ],
            tools=self.args.pydanticTools,
            tool_choice="auto"
        )

        return res


if __name__ == "__main__":
    args = Arguments()
    print(args.domain_to_index)
    print(args.intent_to_index)
    print(args.slot_to_index)
    print(args.slotBIO_to_index)

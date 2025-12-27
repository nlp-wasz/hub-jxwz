"""
使用 Function Calling 实现意图识别 + 领域识别 + 实体识别
基于 Pydantic 模型定义结构化输出
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

# 初始化客户端
client = OpenAI(
    api_key="sk-catsbjffqhpyowokoatelecrweuglhqeehsotwtpleporift",
    base_url="https://api.siliconflow.cn/v1"
)


class ExtractionAgent:
    """通用的结构化信息抽取Agent"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


# ========== 定义实体类型 ==========

class Entity(BaseModel):
    """实体信息"""
    type: str = Field(description="实体类型，如：fromloc, toloc, date, film_name, actor等")
    value: str = Field(description="实体的具体值")
    start: int = Field(description="实体在原文中的起始位置", default=0)
    end: int = Field(description="实体在原文中的结束位置", default=0)


# ========== 定义领域和意图枚举 ==========

class Domain(str, Enum):
    """领域类型"""
    TRAVEL = "Travel"
    VIDEO = "Video"
    FILMTELE = "FilmTele"
    MUSIC = "Music"
    WEATHER = "Weather"
    OTHER = "Other"


class Intent(str, Enum):
    """意图类型（通用）"""
    QUERY = "Query"
    BOOK = "Book"
    CANCEL = "Cancel"
    PLAY = "Play"
    SEARCH = "Search"
    PAUSE = "Pause"
    RECOMMEND = "Recommend"


# ========== 主要的识别模型 ==========

class JointRecognitionResult(BaseModel):
    """联合识别结果：包含领域、意图和实体"""

    domain: Literal["Travel", "Video", "FilmTele", "Music", "Weather", "Other"] = Field(
        description="文本所属领域。Travel:出行交通，Video:视频播放，FilmTele:影视，Music:音乐，Weather:天气，Other:其他"
    )

    intent: Literal["Query", "Book", "Cancel", "Play", "Search", "Pause", "Recommend"] = Field(
        description="用户意图。Query:查询，Book:预订，Cancel:取消，Play:播放，Search:搜索，Pause:暂停，Recommend:推荐"
    )

    entities: List[Entity] = Field(
        description="""从文本中提取的实体列表。
        Travel领域：fromloc(出发地), toloc(目的地), date(日期), vehicle(交通工具)
        Video领域：video_name(视频名), author(作者), category(类别)
        FilmTele领域：film_name(影片名), actor(演员), director(导演), genre(类型)
        Music领域：song_name(歌曲名), artist(歌手), album(专辑)
        Weather领域：location(地点), date(日期)"""
    )


# ========== 简化版本 ==========

def recognize(text: str, model_name: str = "Qwen/QwQ-32B") -> JointRecognitionResult:
    """
    意图识别+领域识别+实体识别

    Args:
        text: 输入文本
        model_name: 使用的模型名称

    Returns:
        JointRecognitionResult: 识别结果
    """
    agent = ExtractionAgent(model_name=model_name)
    prompt = f"请识别以下文本的领域、意图和实体：\n{text}"
    return agent.call(prompt, JointRecognitionResult)


# ========== 测试代码 ==========

if __name__ == "__main__":

    # 测试用例
    test_cases = [
        "还有双鸭山到淮阴的汽车票吗13号的",
        "播放周杰伦的七里香",
        "我想看一部刘德华演的动作片",
        "明天北京的天气怎么样",
        "帮我订一张后天去上海的高铁票",
    ]

    print("=" * 70)
    print("联合识别测试 (Function Calling 方式)")
    print("=" * 70)

    for text in test_cases:
        print(f"\n输入: {text}")
        print("-" * 70)

        result = recognize(text)

        if result:
            print(f"领域: {result.domain}")
            print(f"意图: {result.intent}")
            print(f"实体:")
            for entity in result.entities:
                print(f"  - {entity.type}: {entity.value}")

            # 也可以转为JSON查看
            # print(f"\nJSON输出:\n{result.model_dump_json(indent=2, exclude_unset=True)}")
        else:
            print("识别失败")

        print()

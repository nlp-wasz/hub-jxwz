"""
意图识别 + 领域识别 + 实体识别
使用 OpenAI-compatible API (如 Qwen/QwQ-32B)
"""

from openai import OpenAI

# 初始化客户端（根据你的API配置修改）
client = OpenAI(
    api_key="sk-catsbjffqhpyowokoatelecrweuglhqeehsotwtpleporift",
    base_url="https://api.siliconflow.cn/v1"
)


def joint_recognition(text):
    """
    联合进行意图识别、领域识别和实体识别

    Args:
        text: 输入文本

    Returns:
        dict: 包含领域、意图和实体的结果
    """

    completion = client.chat.completions.create(
        model="Qwen/QwQ-32B",
        messages=[
            {"role": "user", "content": f"""请对下面的文本进行多任务分析，包括：领域识别、意图识别和实体识别。

任务说明：
1. 领域识别（Domain）：判断文本属于哪个领域
   可选领域：Travel（出行）、Video（视频）、FilmTele（影视）、Music（音乐）、Weather（天气）、Other（其他）

2. 意图识别（Intent）：判断用户的具体意图
   Travel领域的意图：Query（查询）、Book（预订）、Cancel（取消）
   Video领域的意图：Play（播放）、Search（搜索）、Pause（暂停）
   FilmTele领域的意图：Play（播放）、Search（搜索）、Recommend（推荐）

3. 实体识别（Entities）：提取文本中的关键实体
   Travel相关：fromloc（出发地）、toloc（目的地）、date（日期）、vehicle（交通工具）
   Video相关：video_name（视频名称）、author（作者）、category（类别）
   FilmTele相关：film_name（影片名称）、actor（演员）、director（导演）、genre（类型）

输入文本：{text}

请按照以下JSON格式输出结果：
{{
  "domain": "领域名称",
  "intent": "意图名称",
  "entities": [
    {{"type": "实体类型", "value": "实体值", "start": 起始位置, "end": 结束位置}}
  ]
}}

只输出JSON，不要包含其他解释。
"""},
        ],
        temperature=0.1,  # 降低温度以获得更稳定的输出
    )

    return completion.choices[0].message.content


def joint_recognition_with_examples(text):
    """
    使用Few-shot learning的方式进行联合识别

    Args:
        text: 输入文本

    Returns:
        dict: 包含领域、意图和实体的结果
    """

    completion = client.chat.completions.create(
        model="Qwen/QwQ-32B",
        messages=[
            {"role": "system", "content": """你是一个专业的NLP助手，擅长进行领域识别、意图识别和实体识别。
请严格按照JSON格式输出结果，不要包含任何额外的解释文字。"""},

            {"role": "user", "content": """输入：还有双鸭山到淮阴的汽车票吗13号的"""},
            {"role": "assistant", "content": """{
  "domain": "Travel",
  "intent": "Query",
  "entities": [
    {"type": "fromloc", "value": "双鸭山", "start": 2, "end": 5},
    {"type": "toloc", "value": "淮阴", "start": 6, "end": 8},
    {"type": "vehicle", "value": "汽车", "start": 9, "end": 11},
    {"type": "date", "value": "13号", "start": 13, "end": 16}
  ]
}"""},

            {"role": "user", "content": """输入：播放周杰伦的七里香"""},
            {"role": "assistant", "content": """{
  "domain": "Music",
  "intent": "Play",
  "entities": [
    {"type": "artist", "value": "周杰伦", "start": 2, "end": 5},
    {"type": "song_name", "value": "七里香", "start": 6, "end": 9}
  ]
}"""},

            {"role": "user", "content": """输入：我想看一部刘德华演的动作片"""},
            {"role": "assistant", "content": """{
  "domain": "FilmTele",
  "intent": "Search",
  "entities": [
    {"type": "actor", "value": "刘德华", "start": 6, "end": 9},
    {"type": "genre", "value": "动作片", "start": 11, "end": 14}
  ]
}"""},

            {"role": "user", "content": f"""输入：{text}"""},
        ],
        temperature=0.1,
    )

    return completion.choices[0].message.content


def joint_recognition_structured(text):
    """
    使用更结构化的提示词格式

    Args:
        text: 输入文本

    Returns:
        str: JSON格式的识别结果
    """

    prompt = f"""# 任务：对用户输入进行联合语义理解

## 输入文本
{text}

## 任务要求

### 1. 领域识别（Domain Classification）
识别文本所属的领域，从以下选项中选择一个：
- Travel: 出行、交通、票务相关
- Video: 视频播放、短视频相关
- FilmTele: 电影、电视剧相关
- Music: 音乐播放、歌曲相关
- Weather: 天气查询相关
- Other: 其他领域

### 2. 意图识别（Intent Detection）
根据领域识别用户的具体意图：

**Travel领域**：
- Query: 查询票务、路线、时间等信息
- Book: 预订车票、机票、酒店等
- Cancel: 取消已有订单

**Video领域**：
- Play: 播放视频
- Search: 搜索视频
- Pause: 暂停播放

**FilmTele领域**：
- Play: 播放影片
- Search: 搜索影片
- Recommend: 推荐影片

**Music领域**：
- Play: 播放音乐
- Search: 搜索歌曲
- Pause: 暂停播放

### 3. 实体识别（Named Entity Recognition）
提取文本中的关键信息实体：

**Travel实体类型**：
- fromloc: 出发地
- toloc: 目的地
- date: 日期时间
- vehicle: 交通工具（火车/汽车/飞机等）

**Video实体类型**：
- video_name: 视频名称
- author: 作者/up主
- category: 视频类别

**FilmTele实体类型**：
- film_name: 影片名称
- actor: 演员
- director: 导演
- genre: 类型（动作/喜剧/爱情等）

**Music实体类型**：
- song_name: 歌曲名称
- artist: 歌手/艺术家
- album: 专辑

## 输出格式
严格按照以下JSON格式输出，不要包含任何其他文字：

{{
  "domain": "领域名称",
  "intent": "意图名称",
  "entities": [
    {{
      "type": "实体类型",
      "value": "实体值",
      "start": 起始字符位置,
      "end": 结束字符位置
    }}
  ]
}}

## 输出
"""

    completion = client.chat.completions.create(
        model="Qwen/QwQ-32B",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    return completion.choices[0].message.content


# 测试示例
if __name__ == "__main__":
    test_cases = [
        "还有双鸭山到淮阴的汽车票吗13号的",
        "播放周杰伦的七里香",
        "我想看一部刘德华演的动作片",
        "明天北京的天气怎么样",
        "帮我订一张后天去上海的高铁票",
    ]

    print("=" * 60)
    print("方法1: 基础提示词")
    print("=" * 60)
    for text in test_cases[:2]:
        print(f"\n输入: {text}")
        result = joint_recognition(text)
        print(f"输出:\n{result}")

    print("\n" + "=" * 60)
    print("方法2: Few-shot Learning")
    print("=" * 60)
    for text in test_cases[:2]:
        print(f"\n输入: {text}")
        result = joint_recognition_with_examples(text)
        print(f"输出:\n{result}")

    print("\n" + "=" * 60)
    print("方法3: 结构化提示词")
    print("=" * 60)
    for text in test_cases[:2]:
        print(f"\n输入: {text}")
        result = joint_recognition_structured(text)
        print(f"输出:\n{result}")

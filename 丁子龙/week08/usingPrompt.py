import openai
import json
import os
client = openai.OpenAI(
    api_key="sk-ea07bf0880504b75a31b1bce38437fcf", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def zero_shot_prompting(text):
    prompt = f"""请对以下用户输入进行语义解析，严格按JSON格式输出，不要包含任何额外文本。

    要求：
    - "intent" 必须是以下之一：OPEN, SEARCH, REPLAY_ALL, NUMBER_QUERY, DIAL, CLOSEPRICE_QUERY, SEND, LAUNCH, PLAY, REPLY, RISERATE_QUERY, DOWNLOAD, QUERY, LOOK_BACK, CREATE, FORWARD, DATE_QUERY, SENDCONTACTS, DEFAULT, TRANSLATION, VIEW, NaN, ROUTE, POSITION
    - "domain" 必须是以下之一：music, app, radio, lottery, stock, novel, weather, match, map, website, news, message, contacts, translation, tvchannel, cinemas, cookbook, joke, riddle, telephone, video, train, poetry, flight, epg, health, email, bus, story
    - "slots" 是一个字典，键只能是以下字段之一：code, Src, startDate_dateOrig, film, endLoc_city, artistRole, location_country, location_area, author, startLoc_city, season, dishNamet, media, datetime_date, episode, teleOperator, questionWord, receiver, ingredient, name, startDate_time, startDate_date, location_province, endLoc_poi, artist, dynasty, area, location_poi, relIssue, Dest, content, keyword, target, startLoc_area, tvchannel, type, song, queryField, awayName, headNum, homeName, decade, payment, popularity, tag, startLoc_poi, date, startLoc_province, endLoc_province, location_city, absIssue, utensil, scoreDescr, dishName, endLoc_area, resolution, yesterday, timeDescr, category, subfocus, theatre, datetime_time
    - 如果没有可提取的槽位，slots 为空字典 {{}}
    - 输出必须是合法 JSON，不要 Markdown，不要解释

    用户输入：{text}

    JSON 输出：
    """
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "user",
             "content": prompt},
        ],
    )
    return completion.choices[0].message.content
# print("\nZero-Shot Prompting")
# print(completion.choices[0].message.content)
def few_shot_prompting(text):
    prompt = """请对以下用户输入进行语义解析，严格按JSON格式输出，不要包含任何额外文本。

    要求：
    - "intent" 必须是以下之一：OPEN, SEARCH, REPLAY_ALL, NUMBER_QUERY, DIAL, CLOSEPRICE_QUERY, SEND, LAUNCH, PLAY, REPLY, RISERATE_QUERY, DOWNLOAD, QUERY, LOOK_BACK, CREATE, FORWARD, DATE_QUERY, SENDCONTACTS, DEFAULT, TRANSLATION, VIEW, NaN, ROUTE, POSITION
    - "domain" 必须是以下之一：music, app, radio, lottery, stock, novel, weather, match, map, website, news, message, contacts, translation, tvchannel, cinemas, cookbook, joke, riddle, telephone, video, train, poetry, flight, epg, health, email, bus, story
    - "slots" 是一个字典，键只能是以下字段之一：code, Src, startDate_dateOrig, film, endLoc_city, artistRole, location_country, location_area, author, startLoc_city, season, dishNamet, media, datetime_date, episode, teleOperator, questionWord, receiver, ingredient, name, startDate_time, startDate_date, location_province, endLoc_poi, artist, dynasty, area, location_poi, relIssue, Dest, content, keyword, target, startLoc_area, tvchannel, type, song, queryField, awayName, headNum, homeName, decade, payment, popularity, tag, startLoc_poi, date, startLoc_province, endLoc_province, location_city, absIssue, utensil, scoreDescr, dishName, endLoc_area, resolution, yesterday, timeDescr, category, subfocus, theatre, datetime_time
    - 如果没有可提取的槽位，slots 为空字典 {{}}
    - 输出必须是合法 JSON，不要 Markdown，不要解释
    
    输入：无锡到阜阳怎么坐汽车？
    输出：    {
        "domain": "bus",
        "intent": "QUERY",
        "slots": {
          "Dest": "阜阳",
          "Src": "无锡"
        }
      }
    
    输入：张绍刚的综艺节目
    输出：   {
        "domain": "video",
        "intent": "QUERY",
        "slots": {
          "artist": "张绍刚",
          "category": "节目",
          "tag": "综艺"
        }
      }""" + f"""
    输入：{text}
    输出："""
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "user",
             "content": prompt},
        ],
    )
    return completion.choices[0].message.content






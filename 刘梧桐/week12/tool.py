import asyncio
import json
import os
from typing import Annotated, Union
import requests
from agents import Agent, Runner
from openai import OpenAI

TOKEN = "738b541a5f7a"

from fastmcp import Client
from fastmcp import FastMCP

mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []


@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []


@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []


@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []


@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []


@mcp.tool
def get_rate_transform(
        source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."],
        aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."],
        money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(
            f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()[
            "result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""

    try:
        # 使用正确的配置 - 针对阿里云百炼
        client = OpenAI(
            api_key="key",  # 替换为阿里云的API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        response = client.chat.completions.create(
            model="qwen-max",  # 或 qwen-plus
            messages=[
                {
                    "role": "system",
                    "content": "你是一个情感分析专家。请分析文本情感，只返回JSON格式：{'sentiment': 'positive/negative/neutral', 'confidence': 0.0-1.0, 'reason': '简要分析原因'}"
                },
                {
                    "role": "user",
                    "content": f"分析以下文本的情感：{text}"
                }
            ],
            temperature=0.1
        )

        result = response.choices[0].message.content
        # 解析JSON结果
        try:
            return json.loads(result)
        except:
            return {"result": result}
    except:
        return "Unable to analyze sentiment at the moment"
async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)
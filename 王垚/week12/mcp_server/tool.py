"""通用工具类 MCP Server，汇总天气、地址解析、电话归属地、花语、汇率等接口。"""
import logging

from typing import Annotated, Union
import re
import requests
from fastmcp import FastMCP
logger = logging.getLogger(__name__)

TOKEN = "738b541a5f7a"

# 创建工具类 FastMCP Server，对外暴露多个实用 API
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""Utility APIs such as weather, address parsing, telecom lookup, etc.""",
)


@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """查询城市（拼音）对应的实时天气。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_city_weather(city_name={city_name})")
        resp = requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}", timeout=5)
        return resp.json().get("data", {})
    except Exception:
        return {}


@mcp.tool
def get_address_detail(address_text: Annotated[str, "Raw address text in Chinese"]):
    """解析原始地址文本，返回省市区等结构化信息。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_address_detail(address_text={address_text[:20]}...)")
        resp = requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}", timeout=5)
        return resp.json().get("result", {})
    except Exception:
        return {}


@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Phone number to look up"]):
    """查询手机号归属地、运营商等信息。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_tel_info(tel_no={tel_no})")
        resp = requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}", timeout=5)
        return resp.json().get("result", {})
    except Exception:
        return {}


@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Name of scenic spot or tourist attraction"]):
    """根据景点名称返回地点介绍、所属城市等数据。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_scenic_info(scenic_name={scenic_name})")
        resp = requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}", timeout=5)
        data = resp.json()
        return data.get("result", {}).get("list", [])
    except Exception:
        return []


@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name in Chinese"]):
    """查询花语、寓意等花卉相关信息。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_flower_info(flower_name={flower_name})")
        resp = requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}", timeout=5)
        return resp.json().get("result", {})
    except Exception:
        return {}


@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."],
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."],
    money: Annotated[Union[int, float], "The amount of money to convert."],
):
    """通过汇率 API 计算不同币种的兑换金额。"""
    try:
        logger.info(f"[MCP][tools] 调用 get_rate_transform({source_coin}->{aim_coin}, money={money})")
        resp = requests.get(
            f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}",
            timeout=5,
        )
        return resp.json().get("result", {}).get("money", None)
    except Exception:
        return None


POSITIVE_WORDS = {
    "happy", "great", "excellent", "good", "awesome", "love", "fantastic", "positive",
    "满意", "开心", "棒", "喜欢", "赞", "不错", "满意度高",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "sad", "angry", "hate", "poor", "negative",
    "糟糕", "失望", "生气", "讨厌", "垃圾", "不好", "差",
}


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """对输入文本进行情感分类（正向/反向/中性），同时返回中英文两套标签。"""
    normalized = text.lower()
    # 保留中文及字母数字，其余字符替换为空格，方便统计词频
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)

    def _score(words: set[str]) -> int:
        return sum(normalized.count(w.lower()) for w in words)

    positive_score = _score(POSITIVE_WORDS)
    negative_score = _score(NEGATIVE_WORDS)

    # 英文标签：便于程序内部使用
    # 中文标签：与 Week05 / Week08 示例中“正向/反向”风格保持一致
    if positive_score > negative_score:
        sentiment = "positive"
        sentiment_zh = "正向"
    elif negative_score > positive_score:
        sentiment = "negative"
        sentiment_zh = "反向"
    else:
        sentiment = "neutral"
        sentiment_zh = "中性"

    # 返回结构化结果，方便 Agent 解释或进一步处理
    return {
        "sentiment": sentiment,
        "sentiment_zh": sentiment_zh,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "text": text,
    }

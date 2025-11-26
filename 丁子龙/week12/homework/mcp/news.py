import requests

from fastmcp import FastMCP
from typing import List, Dict, Any, Union,Annotated
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server contains some api of news.""",
)
TOKEN = "738b541a5f7a"
supported_sites = {
    "百度": "baidu",
    "少数派": "shaoshupai",
    "微博": "weibo",
    "知乎": "zhihu",
    "36氪": "36kr",
    "吾爱破解": "52pojie",
    "哔哩哔哩": "bilibili",
    "豆瓣": "douban",
    "虎扑": "hupu",
    "贴吧": "tieba",
    "掘金": "juejin",
    "抖音": "douyin",
    "V2EX": "v2ex",
    "今日头条": "jinritoutiao"
}
platforms = list(supported_sites.values())
# print( platforms)
@mcp.tool
async def getNews(platform:  Annotated[str, f"The Pinyin of the platform name (e.g., {platforms})"]) -> str:
    """Retrieves news from a specific platform."""
    url = "https://orz.ai/api/v1/dailynews/"
    if platform in platforms:
        params = {
            "platform": platform
        }
        try:
            return requests.get(url, params=params).json()["data"]
        except:
            return []

@mcp.tool
def get_today_daily_news():
    """Retrieves a list of today's daily news bulletin items from the external API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/bulletin?key={TOKEN}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_douyin_hot_news():
    """Retrieves a list of trending topics or hot news from Douyin (TikTok China) using the API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/douyinhot?key={TOKEN}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_github_hot_news():
    """Retrieves a list of trending repositories/projects on GitHub using the API."""
    try:
        return requests.get(f"https://whyta.cn/api/github?key={TOKEN}").json()["items"]
    except:
        return []

@mcp.tool
def get_toutiao_hot_news():
    """Retrieves a list of hot news headlines from Toutiao (a Chinese news platform) using the API."""
    try:
        print(f"https://whyta.cn/api/tx/topnews?key={TOKEN}")
        return requests.get(f"https://whyta.cn/api/tx/topnews?key={TOKEN}").json()["result"]["list"]
    except:
        import traceback
        traceback.print_exc()
        return []

@mcp.tool
def get_sports_news():
    """Retrieves a list of esports or general sports news items using the external API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/esports?key={TOKEN}").json()["result"]["newslist"]
    except:
        return []
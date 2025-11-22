"""语录/心灵鸡汤类 MCP 工具，提供每日金句服务。"""
import logging

import requests
from fastmcp import FastMCP
logger = logging.getLogger(__name__)
TOKEN = "738b541a5f7a"

# FastMCP Server：对外暴露语录、励志类工具
mcp = FastMCP(
    name="Saying-MCP-Server",
    instructions="""Quote and saying related MCP tools.""",
)


@mcp.tool
def get_today_famous_saying():
    """获取一句今日随机金句/名言。"""
    try:
        logger.info("[MCP][saying] 调用 get_today_famous_saying() 获取随机金句")
        resp = requests.get(f"https://whyta.cn/api/yiyan?key={TOKEN}", timeout=5)
        data = resp.json()
        return data.get("hitokoto", "")
    except Exception:
        # 返回空字符串，保持接口稳定
        return ""


@mcp.tool
def get_today_motivation_saying():
    """获取励志语录，可用于鼓励或每日打气。"""
    try:
        logger.info("[MCP][saying] 调用 get_today_motivation_saying() 获取励志语录")
        resp = requests.get(f"https://whyta.cn/api/tx/lzmy?key={TOKEN}", timeout=5)
        data = resp.json()
        return data.get("result", {})
    except Exception:
        return {}

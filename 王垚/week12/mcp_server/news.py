"""新闻类 MCP 工具，提供不同热点资讯的查询能力。"""

import logging

import requests
from fastmcp import FastMCP


logger = logging.getLogger(__name__)

TOKEN = "738b541a5f7a"

# 创建 FastMCP Server，用于对外暴露新闻相关工具
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""News related APIs exposed as MCP tools.""",
)


@mcp.tool
def get_today_daily_news():
    """查询当日新闻简报列表。"""
    try:
        logger.info("[MCP][news] 调用 get_today_daily_news() 查询今日新闻简报")
        resp = requests.get(f"https://whyta.cn/api/tx/bulletin?key={TOKEN}", timeout=5)
        data = resp.json()
        return data.get("result", {}).get("list", [])
    except Exception:
        # 对外暴露工具时，出现异常保持接口幂等，返回空列表
        return []


@mcp.tool
def get_douyin_hot_news():
    """查询抖音热榜，返回当前热门话题。"""
    try:
        logger.info("[MCP][news] 调用 get_douyin_hot_news() 查询抖音热榜")
        resp = requests.get(f"https://whyta.cn/api/tx/douyinhot?key={TOKEN}", timeout=5)
        data = resp.json()
        return data.get("result", {}).get("list", [])
    except Exception:
        return []

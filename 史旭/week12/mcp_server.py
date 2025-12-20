# 将 news_mcp、saying_mcp、tool_mcp 整合到当前 MCPServer 模块中，启动MCP后端服务模块
import asyncio

from news_mcp import mcp as news
from saying_mcp import mcp as saying
from tool_mcp import mcp as tool
from emotion_mcp import mcp as emotion

from fastmcp import FastMCP

mcp = FastMCP(name="mcp_tools", instructions="整合所有后端tools")


# 导入 其它模块 mcp tool
async def import_mcp_server():
    await asyncio.gather(
        mcp.import_server(server=news, prefix="news"),
        mcp.import_server(server=saying, prefix="saying"),
        mcp.import_server(server=tool, prefix="tool"),
        mcp.import_server(server=emotion, prefix="emotion"),
    )


# 根据 工具类型 选择对应的mcp工具（不管用，没找到问题所在）
async def byMcpTypeGetTools(mcp_type):
    # 获取 mcp中的所有工具
    mcp_type_tools = await mcp.get_tools()
    if mcp_type == "all":
        return mcp_type_tools
    return [tool_name for tool_name in mcp_type_tools if tool_name.startswith(f"{mcp_type}_")]


# 根据 工具类型 选择对应的mcp工具（手动）
def byMcpTypeGetToolsAuto(mcp_type):
    # 自定义所有mcp_tools
    mcp_type_tools = ['news_get_today_daily_news', 'news_get_douyin_hot_news', 'news_get_github_hot_news',
                      'news_get_toutiao_hot_news', 'news_get_sports_news', 'saying_get_today_familous_saying',
                      'saying_get_today_motivation_saying', 'saying_get_today_working_saying', 'tool_get_city_weather',
                      'tool_get_address_detail', 'tool_get_tel_info', 'tool_get_scenic_info', 'tool_get_flower_info',
                      'tool_get_rate_transform', 'emotion_sentiment_classification']
    if mcp_type == "all":
        return mcp_type_tools
    return [tool_name for tool_name in mcp_type_tools if tool_name.startswith(f"{mcp_type}_")]


if __name__ == '__main__':
    asyncio.run(import_mcp_server())
    mcp.run(transport="sse", port=8000)

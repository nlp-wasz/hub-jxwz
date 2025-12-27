import asyncio
from fastmcp import FastMCP, Client

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from emotion import mcp as emotion_mcp

mcp = FastMCP(
    name="MCP-Server"
)


# 定义工具过滤器
def tool_filter(tools, context):
    """
    工具过滤器，根据上下文决定可用的工具
    """
    # 如果没有上下文或者没有特定要求，则返回所有工具
    if not context or 'category' not in context:
        return tools

    category = context['category']

    # 根据类别过滤工具
    if category == 'news':
        # 只返回新闻相关的工具
        return [tool for tool in tools if tool.name.startswith('news')]
    elif category == 'tool':
        # 只返回工具相关的工具
        return [tool for tool in tools if tool.name.startswith('tool')]
    elif category == 'saying':
        # 只返回名言相关的工具
        return [tool for tool in tools if tool.name.startswith('saying')]
    elif category == 'emotion':
        # 只返回情感相关的工具
        return [tool for tool in tools if tool.name.startswith('emotion')]

    # 默认返回所有工具
    return tools

async def setup():
    await mcp.import_server(news_mcp, prefix="")
    await mcp.import_server(saying_mcp, prefix="")
    await mcp.import_server(tool_mcp, prefix="")
    await mcp.import_server(emotion_mcp, prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)

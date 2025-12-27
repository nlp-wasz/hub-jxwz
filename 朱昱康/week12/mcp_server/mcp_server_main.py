import asyncio
from fastmcp import FastMCP, Client

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from sentiment_analysis import mcp as sentiment_mcp

# 创建不同类别的MCP服务器
news_mcp_server = FastMCP(name="News-MCP-Server")
tools_mcp_server = FastMCP(name="Tools-MCP-Server")
saying_mcp_server = FastMCP(name="Saying-MCP-Server")
sentiment_mcp_server = FastMCP(name="Sentiment-MCP-Server")

# 主服务器，包含所有工具
mcp = FastMCP(name="MCP-Server")

async def setup():
    # 设置分类服务器
    await news_mcp_server.import_server(news_mcp, prefix="")
    await tools_mcp_server.import_server(tool_mcp, prefix="")
    await saying_mcp_server.import_server(saying_mcp, prefix="")
    await sentiment_mcp_server.import_server(sentiment_mcp, prefix="")
    
    # 设置主服务器，包含所有工具
    await mcp.import_server(news_mcp, prefix="")
    await mcp.import_server(saying_mcp, prefix="")
    await mcp.import_server(tool_mcp, prefix="")
    await mcp.import_server(sentiment_mcp, prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

async def run_news_server():
    """运行新闻服务器"""
    await setup()
    news_mcp_server.run(transport="sse", port=8901)

async def run_tools_server():
    """运行工具服务器"""
    await setup()
    tools_mcp_server.run(transport="sse", port=8902)

async def run_saying_server():
    """运行名言服务器"""
    await setup()
    saying_mcp_server.run(transport="sse", port=8903)

async def run_sentiment_server():
    """运行情感分析服务器"""
    await setup()
    sentiment_mcp_server.run(transport="sse", port=8904)

if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    
    # 默认运行主服务器
    mcp.run(transport="sse", port=8900)

import asyncio

from fastmcp import FastMCP, Client

from mcp_server.news import mcp as news_mcp
from mcp_server.saying import mcp as saying_mcp
from mcp_server.tool import mcp as tool_mcp


mcp = FastMCP(
    name="MyJob-MCP-Server",
)


async def setup() -> None:
    """注册并整合子 MCP Server（新闻/语录/工具）到一个根 Server 中。"""
    await mcp.import_server(news_mcp, prefix="news-")
    await mcp.import_server(saying_mcp, prefix="saying-")
    await mcp.import_server(tool_mcp, prefix="tool-")


async def test_list_tools() -> None:
    """启动时打印当前可用的工具列表。"""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_list_tools())
    # 启动 SSE 服务，监听 8900 端口，供 Streamlit 侧的 MCPServerSse 连接
    mcp.run(transport="sse", port=8900)

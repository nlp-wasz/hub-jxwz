# 主 MCP 客户端模块
import asyncio

from fastmcp import FastMCP, Client
from fast_api.StockApiServer import stock_api
from mcp_client.NewsMcp import mcp as news_mcp
from mcp_client.SayingMcp import mcp as saying_mcp
from mcp_client.ToolMcp import mcp as tool_mcp

mcp = FastMCP.from_fastapi(stock_api)


# 挂载其它mcp服务
async def import_server():
    await mcp.import_server(news_mcp)
    await mcp.import_server(saying_mcp)
    await mcp.import_server(tool_mcp)


# 获取 mcp_client 工具列表
async def get_mcp_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tools_name = [tool.name for tool in tools]

        print(f"tool_name：{tools_name}")


if __name__ == '__main__':
    asyncio.run(import_server())
    asyncio.run(get_mcp_tools())
    mcp.run(transport="sse", port=8002)

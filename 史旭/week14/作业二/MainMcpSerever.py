# MCP 服务端
import asyncio

from fastmcp import FastMCP, Client
from mcp_tools_1 import mcp as mcp1
from mcp_tools_2 import mcp as mcp2

mcp = FastMCP("MCP 服务端")


async def load_mcp():
    await mcp.import_server(mcp1)
    await mcp.import_server(mcp2)


async def tool_names():
    async with Client(mcp) as client:
        tools = await client.list_tools()

        print([tool_name for tool_name in tools])


if __name__ == "__main__":
    asyncio.run(load_mcp())
    asyncio.run(tool_names())

    mcp.run(transport="sse", port=8000)

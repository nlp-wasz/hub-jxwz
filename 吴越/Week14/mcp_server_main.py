from task_tool import mcp as tool_mcp
from fastmcp import FastMCP, Client
import asyncio

mcp = FastMCP(
    name="MCP-Server"
)

async def setup():

    await mcp.import_server(tool_mcp, prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)
import asyncio
import requests  # type: ignore
from fastmcp import FastMCP, Client

from api.autostock import app
from api.news import mcp as news_mcp
from api.saying import mcp as saying_mcp
from api.tool import mcp as tool_mcp

mcp = FastMCP.from_fastapi(app=app)

async def setup():
    await mcp.import_server(news_mcp, prefix="")
    await mcp.import_server(saying_mcp, prefix="")
    await mcp.import_server(tool_mcp, prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print("Available tools:", [t for t in tools])

if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)

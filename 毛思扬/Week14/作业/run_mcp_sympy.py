import asyncio
from fastmcp import FastMCP, Client

from mcp_sympy_tools import mcp as sympy_mcp

mcp = FastMCP()


async def setup():
    await mcp.import_server(sympy_mcp, prefix="")


async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print("Available tools:", [t for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)

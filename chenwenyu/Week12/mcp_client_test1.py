import asyncio
from fastmcp import Client

client = Client("http://localhost:8900/sse")

async def call_tool():
    # async with client:
    #     result = await client.call_tool("get_today_daily_news")
    #     print(result)

    async with client:
        result = await client.list_tools()
        print(result)

        result = await client.call_tool("get_today_motivation_saying")
        print(result)

        result = await client.call_tool("get_city_weather",{"city_name":"beijing"})
        print(result)

        result = await client.call_tool("text_classification",{"text":"今天的乒乓球比赛特别精彩"})
        print(result)

asyncio.run(call_tool())

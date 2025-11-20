import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-fad1550b59d547ee83006bde2452e7bc"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


async def run(mcp_server: MCPServer):
    available_tools = await mcp_server.list_tools()
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    agent = Agent(
        name="Assistant",
        instructions="根据用户需求调用合适的工具",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-flash",
            openai_client=external_client,
        )
    )

    message = "分类这段文本：今天的天气太让人沮丧了！"
    print(f"Running: {message}\n")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


    message = "合肥最近的天气怎么样？"
    print(f"\nRunning: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


async def main():
    #创建MCP的Server
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
    )as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(main())

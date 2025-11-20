import os
from typing import Any, Iterable, Optional, List

from agents.util._types import MaybeAwaitable

'''
openai方式调用
'''
# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-4c44ef4112a04e65910dfdd56774f084"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer, ToolFilterContext
from agents import set_default_openai_api, set_tracing_disabled
from mcp.types import Tool as MCPTool

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

def tool_filter(context: ToolFilterContext, tool: MCPTool) -> MaybeAwaitable[bool]:
    """
    工具过滤器函数，根据工具名称筛选工具

    Args:
        context: 工具过滤上下文
        tool: MCP工具对象

    Returns:
        bool: True表示允许使用该工具，False表示过滤掉该工具
    """
    print(context.run_context.context)    
    try:
        # 获取工具名称
        tool_name = tool.name.lower() if tool.name else ""

        # 根据需求实现过滤逻辑：
        # 1. 查询新闻的时候，只调用包含'news'的工具
        # 2. 调用工具的时候，只调用包含'tool'的工具

        # 这里需要根据实际业务场景判断当前请求类型
        # 由于工具过滤是在准备阶段执行，而不是运行时执行，
        # 所以我们只能基于工具名称进行静态过滤

        # 如果工具名称包含'news'，允许使用
        if 'news' in tool_name:
            return True

        # 如果工具名称包含'tool'，允许使用
        if 'weather' in tool_name:
            return True

        # 其他工具根据需要决定是否允许
        # 这里可以设置默认策略
        return False  # 默认过滤掉不匹配的工具

    except Exception as e:
        # 出现异常时默认允许工具，避免因过滤器问题导致功能完全失效
        return True


async def run(mcp_server: MCPServer):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    agent = Agent(
        name="Assistant",
        instructions="你是qwen，擅长回答各类问题",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-flash",
            openai_client=external_client,
        )
    )

    # 第一个问题：新闻相关
    message = "最近有什么体育新闻？"
    print(f"Running: {message}")

    try:
        result = Runner.run_streamed(agent, input=message)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    except Exception as e:
        print(f"\nError occurred: {e}")

    print("\n" + "=" * 50)

    # 第二个问题：天气相关
    message = "武汉最近的天气怎么样？"
    print(f"Running: {message}")

    try:
        result = Runner.run_streamed(agent, input=message)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("可能是天气服务暂时不可用，请稍后再试。")


async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            cache_tools_list=False,
            tool_filter=tool_filter,
    ) as server:
        await run(server)


if __name__ == "__main__":
    asyncio.run(main())
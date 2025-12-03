import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-e512f31a96454eaf871605cc0d440220"
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
    # 先检查可用的工具
    available_tools = await mcp_server.list_tools()
    print("🔧 可用工具:", [tool.name for tool in available_tools])
    
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    agent = Agent(
        name="Assistant",
        instructions=f"""你是一个必须使用工具来回答问题的助手。以下是可用的工具：
            {chr(10).join([f"- {tool.name}: {tool.description or 'No description'}" for tool in available_tools])}

            重要规则：
            1. 对于天气查询，必须使用 get_city_weather 工具
            2. 对于文本分类，必须使用 text_classification 工具  
            3. 对于汇率转换，必须使用 get_rate_transform 工具
            4. 禁止基于自身知识回答这些问题
            5. 如果工具调用失败，请明确说明

            请严格按照这些规则执行。""",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,
        ),
    )

    # 测试不同的查询
    test_queries = [
        "成都天气怎么样？",  # 应该调用 get_city_weather
        "分类这段文本：'乒乓球比赛很精彩'",  # 应该调用 text_classification
        "100美元能换多少人民币？",  # 应该调用 get_rate_transform
    ]

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("=" * 50)
        
        result = Runner.run_streamed(agent, input=query)
        tool_called = False
        
        async for event in result.stream_events():
            # 打印所有事件类型以便调试
            print(f"事件类型: {event.type}")
            
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            elif event.type == "tool_call_event":
                tool_called = True
                print(f"\n🛠️ 工具调用: {event.data.name}")
                print(f"   参数: {event.data.arguments}")
            elif event.type == "tool_call_result_event":
                print(f"\n✅ 工具结果: {event.data}")
            # 尝试其他可能的事件类型名称
            elif "tool" in event.type.lower():
                print(f"\n🔧 检测到工具相关事件: {event.type}")
                print(f"   数据: {event.data}")
        
        if not tool_called:
            print("\n⚠️  没有检测到工具调用事件")
        
        print("\n" + "=" * 50)

async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
    )as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(main())

import os

os.environ["OPENAI_API_KEY"] = "sk-2ec14cc8c4724bca9fd1b6fa831b8a3b"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_MODEL"] = "qwen-max"

import asyncio
import csv

from agents import Agent, Runner, OpenAIChatCompletionsModel

from agents.mcp import MCPServerSse, MCPServer
from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncOpenAI
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)
external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


async def read_questions():
    """读取问题列表"""
    questions = []
    with open('questions.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['question'])
    return questions


async def run_mcp_numpy(mcp_server: MCPServer):
    questions = await read_questions()

    numpy_agent = Agent(
        name="numpy_agent",
        instructions="",
        mcp_servers=[mcp_server],
        tool_use_behavior="run_llm_again",
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        )
    )

    for question in questions:
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        
        result = Runner.run_streamed(numpy_agent, input=question)

        assistant_message = ""
        async for event in result.stream_events():
            # run llm again 的回答： 基础tool的结果继续回答
            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                            ResponseTextDeltaEvent):
                print(event.data.delta, end="")
                assistant_message += event.data.delta
        print()  # 换行


async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8901/sse",
            },
            client_session_timeout_seconds=20
    ) as mcp_server:
        await run_mcp_numpy(mcp_server)


if __name__ == "__main__":
    print("请确保已经运行了 run_mcp_numpy.py 服务...")
    print("可以通过以下命令启动服务:")
    print("python run_mcp_numpy.py")
    
    # 用户确认服务已启动后再继续
    input("确认服务已启动后，按回车键继续...")
    
    asyncio.run(main())
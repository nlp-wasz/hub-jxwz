import os

from agents.extensions.memory import AdvancedSQLiteSession
from openai import AsyncOpenAI

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-399b434c3f5b4329a4600ec76ce4f7cc"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_MODEL"] = "qwen-max"
os.environ["OPENAI_VISON_MODEL"] = "qwen-vl"

import asyncio

import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def main():
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    session = AdvancedSQLiteSession(
        session_id="test1",
        db_path="./assert/conversations.db",
        create_tables=True
    )

    agent = Agent(
        name="Assistant",
        instructions="你好，你是小王",
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
    )

    result = Runner.run_streamed(agent, input="你叫什么名字", session=session)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    result = Runner.run_streamed(agent, input="我之前的问题是什么？", session=session)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

asyncio.run(main())

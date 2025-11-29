import os
from typing import AsyncGenerator, Optional, List
from enum import Enum

from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI


class HandoffAgentSystem:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )

    def _create_router_agent(self, chat_agent: Agent, stock_agent: Agent) -> Agent:
        """创建路由分发Agent"""
        instructions = """
你是一个智能路由助手，负责分析用户输入并决定由哪个专业助手处理。

请根据以下规则进行路由：
- 如果用户询问股票、股价、投资、财报、K线等金融相关话题，转给股票助手
- 如果用户询问日常聊天、天气、笑话、故事等闲聊话题，转给闲聊助手
- 如果无法确定，默认转给闲聊助手

请只返回路由决策，不要回答问题本身。
"""
        return Agent(
            name="Router",
            instructions=instructions,
            handoffs=[chat_agent, stock_agent],  # 关键：使用handoffs参数
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.client,
            )
        )

    def _create_chat_agent(self) -> Agent:
        """创建闲聊Agent"""
        instructions = """
你是一个友好的闲聊助手，专注于日常对话。
保持轻松愉快的语气，提供情感支持。
回答要简洁富有情感。
"""
        return Agent(
            name="ChatAssistant",
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.client,
            )
        )

    def _create_stock_agent(self) -> Agent:
        """创建股票分析Agent"""
        instructions = """
你是一个专业的股票分析助手，专注于股票市场分析。
使用专业金融术语，如P/E、EPS、ROI等。
提供分析时必须说明数据来源和局限性，强调不构成投资建议。
"""
        return Agent(
            name="StockAnalyst",
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.client,
            )
        )

    async def chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """使用handoff机制的主对话函数"""
        # 创建专业Agent
        chat_agent = self._create_chat_agent()
        stock_agent = self._create_stock_agent()

        # 创建路由Agent，并设置handoffs
        router_agent = self._create_router_agent(chat_agent, stock_agent)

        # 运行路由Agent，它会自动handoff到合适的专业Agent
        result = Runner.run_streamed(router_agent, input=user_input)

        async for event in result.stream_events():
            if hasattr(event, 'data') and hasattr(event.data, 'delta'):
                yield event.data.delta


# 使用测试
async def main():
    system = HandoffAgentSystem()

    # 测试不同场景
    test_messages = [
        "你好，今天天气怎么样？",
        "我想了解苹果公司的股票情况",
        "讲个笑话吧",
        "帮我分析一下腾讯的财报"
    ]

    for msg in test_messages:
        print(f"用户: {msg}")
        print("助手: ", end="")

        async for response in system.chat(msg):
            print(response, end="", flush=True)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
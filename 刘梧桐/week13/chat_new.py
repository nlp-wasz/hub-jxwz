import asyncio
import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, Handoff
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseOutputItemDoneEvent,
    ResponseFunctionToolCall
)
from jinja2 import Environment, FileSystemLoader

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse


# 生成会话ID的工具函数（保持不变）
def generate_random_chat_id(length=12):
    with SessionLocal() as session:
        for retry_time in range(20):
            characters = string.ascii_letters + string.digits
            session_id = ''.join(random.choice(characters) for i in range(length))
            chat_session_record: ChatSessionTable | None = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id).first()
            if chat_session_record is None:
                break

            if retry_time > 10:
                raise Exception("Failed to generate a unique session_hash")

    return session_id


# 初始化消息模板（扩展多Agent支持）
def get_agent_instructions(agent_type: str) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinjia2")

    if agent_type == "stock":
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
6. 当检测到用户话题切换为非股票内容时，使用`handoff`工具切换到闲聊Agent
"""
    elif agent_type == "chat":
        task_description = """
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
5. 关键词：友好、轻松、富有同理心。
6. 当检测到用户话题涉及股票相关内容时，使用`handoff`工具切换到股票Agent
"""

    return template.render(
        agent_name="小呆助手" if agent_type == "chat" else "股票分析助手",
        task_description=task_description,
        current_datetime=datetime.now(),
    )


# 初始化会话（保持不变）
def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        initial_agent: str = "chat",  # 默认从闲聊Agent开始
) -> str:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        session.add(chat_session_record)
        session.commit()
        session.flush()

        # 存储初始Agent类型的系统消息
        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_agent_instructions(initial_agent)
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


# 新增：消息存储工具函数
def append_message2db(session_id: str, role: str, content: str):
    with SessionLocal() as session:
        chat_session = session.query(ChatSessionTable).filter(
            ChatSessionTable.session_id == session_id
        ).first()
        if chat_session:
            message = ChatMessageTable(
                chat_id=chat_session.id,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()


# 核心对话处理函数（支持Agent切换）
async def chat(
        user_name: str,
        session_id: Optional[str],
        content: str,
        initial_agent: str = "chat",
        tools: List[str] = []
) -> AsyncGenerator[str, None]:
    # 会话管理
    if not session_id:
        session_id = generate_random_chat_id()
        init_chat_session(user_name, content, session_id, initial_agent)
    else:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id
            ).first()
            if not record:
                init_chat_session(user_name, content, session_id, initial_agent)

    # 存储用户消息
    append_message2db(session_id, "user", content)

    # 初始化OpenAI客户端
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # 配置MCP工具
    tool_filter = ToolFilterStatic(allowed_tool_names=tools) if tools else None
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_filter,
        client_session_timeout_seconds=20,
    )

    # 会话存储
    session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 定义股票Agent
    def create_stock_agent():
        return Agent(
            name="StockAgent",
            instructions=get_agent_instructions("stock"),
            mcp_servers=[mcp_server] if tools else [],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            ),
            tool_use_behavior="stop_on_first_tool" if any(
                t in tools for t in ["get_day_line", "get_week_line"]) else "run_llm_again",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

    # 定义闲聊Agent
    def create_chat_agent():
        return Agent(
            name="ChatAgent",
            instructions=get_agent_instructions("chat"),
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

    # 初始Agent
    current_agent = create_chat_agent() if initial_agent == "chat" else create_stock_agent()
    assistant_message = ""

    # 处理Agent切换的流式响应
    async with mcp_server if tools else asyncio.NoContextManager():
        result = Runner.run_streamed(current_agent, input=content, session=session)

        async for event in result.stream_events():
            # 处理Agent切换信号
            if event.type == "raw_response_event" and isinstance(event.data, ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall) and event.data.item.name == "handoff":
                    # 解析切换目标
                    handoff_params = event.data.item.arguments
                    target_agent = handoff_params.get("target", "chat")

                    # 发送切换提示
                    switch_msg = f"\n[已切换至{('股票分析助手' if target_agent == 'stock' else '闲聊助手')}]\n"
                    yield switch_msg
                    assistant_message += switch_msg

                    # 切换Agent并继续处理
                    current_agent = create_stock_agent() if target_agent == "stock" else create_chat_agent()
                    result = Runner.run_streamed(current_agent, input=content, session=session)
                    continue

            # 处理普通文本响应
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                if event.data.delta:
                    yield event.data.delta
                    assistant_message += event.data.delta

            # 处理工具调用响应
            if event.type == "raw_response_event" and isinstance(event.data, ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    tool_msg = f"\n```json\n{event.data.item.name}:{event.data.item.arguments}\n```\n\n"
                    yield tool_msg
                    assistant_message += tool_msg

    # 存储Agent回复
    append_message2db(session_id, "assistant", assistant_message)
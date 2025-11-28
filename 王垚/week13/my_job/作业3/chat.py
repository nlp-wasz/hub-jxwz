import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from typing import AsyncGenerator

from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse


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


def _make_model():
    return OpenAIChatCompletionsModel(
        model=os.environ["OPENAI_MODEL"],
        openai_client=AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        ),
    )


stock_agent = Agent(
    name="stock_agent",
    instructions=(
        """
        你是金融专家你是数据分析师你现在要处理的事情有:
        1、专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
        2、必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
        3、在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议
        4、仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
        5、结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
        """
    ),
    model=_make_model(),
)

bi_agent = Agent(
    name="bi_agent",
    instructions=(
        """
        你是数据分析师你现在要处理的事情有:
        1. 帮助用户理解他们的数据结构、商业指标和关键绩效指标 (KPI)。
        2. 用户的请求通常是数据查询、指标定义或图表生成建议。
        3. **关键约束：你的输出必须是可执行的代码块 (如 SQL 或 Python)，或者清晰的逻辑步骤，用于解决用户的数据问题。**
        4. 严格遵守数据分析的逻辑严谨性，确保每一个结论都有数据支撑。
        5. 当被要求提供可视化建议时，请推荐最合适的图表类型（如：时间序列用折线图，分类对比用柱状图）。
        """
    ),
    model=_make_model(),
)

casual_agent = Agent(
    name="casual_agent",
    instructions=(
        """
        你是友好、轻松的聊天助手,你要做的事情有:
        1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
        2. 避免过于专业或生硬的术语，除非用户明确要求。
        3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
        4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
        5. 关键词：友好、轻松、富有同理心。
        """
    ),
    model=_make_model(),
)

# Triage Agent：根据用户输入自动选择子 Agent
triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "请根据用户的问题内容，判断应由哪个专家处理：\n"
        "- 如果涉及股票、金融、投资、财报、估值等，请 handoff 给 stock_agent；\n"
        "- 如果涉及数据分析、SQL、BI、KPI、可视化、指标计算等，请 handoff 给 bi_agent；\n"
        "- 其他日常对话、闲聊、非专业问题，请 handoff 给 casual_agent。\n"
        "优先使用 handoff，不要自行回答专业问题。"
    ),
    model=_make_model(),
    handoffs=[stock_agent, bi_agent, casual_agent],
)


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
) -> str:
    # 创建对话的title，通过summary agent
    # 存储数据库
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        print("add ChatSessionTable", user_id[0], session_id)
        session.add(chat_session_record)
        session.commit()
        session.flush()
    return True


async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # agent 初始化
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # mcp tools 选择
    if not tools or len(tools) == 0:
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
        tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id,  # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 如果没有选择工具，默认直接调用大模型回答
    if tools:
        # 动态创建带 MCP 的 triage agent
        need_viz_tools = {"get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"}
        tool_use_behavior = "stop_on_first_tool" if set(tools) & need_viz_tools else "run_llm_again"

        runnable_agent = Agent(
            name=triage_agent.name,
            instructions=triage_agent.instructions,
            model=triage_agent.model,
            handoffs=triage_agent.handoffs,
            mcp_servers=[mcp_server],
            tool_use_behavior=tool_use_behavior,
            model_settings=ModelSettings(parallel_tool_calls=False)
        )
    else:
        runnable_agent = triage_agent
    if tools:
        async with mcp_server:
            result = Runner.run_streamed(runnable_agent, input=content, session=agent_session)
    else:
        result = Runner.run_streamed(runnable_agent, input=content, session=agent_session)

    assistant_message = ""
    async for event in result.stream_events():
        # 处理工具调用（显示工具名和参数）
        if (
                event.type == "raw_response_event"
                and isinstance(event.data, ResponseOutputItemDoneEvent)
                and isinstance(event.data.item, ResponseFunctionToolCall)
        ):
            tool_repr = f"\n```json\n{event.data.item.name}:{event.data.item.arguments}\n```\n\n"
            yield tool_repr
            assistant_message += tool_repr

        # 处理文本输出
        elif (
                event.type == "raw_response_event"
                and isinstance(event.data, ResponseTextDeltaEvent)
                and event.data.delta
        ):
            yield event.data.delta
            assistant_message += event.data.delta

    # 保存助手回复
    append_message2db(session_id, "assistant", assistant_message)


def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:

        chat_messages: Optional[List[ChatMessageTable]] = session.query(ChatMessageTable) \
            .join(ChatSessionTable) \
            .filter(
            ChatSessionTable.session_id == session_id
        ).all()

        result = []
        if chat_messages:
            for record in chat_messages:
                result.append({
                    "id": record.id, "create_time": record.create_time,
                    "feedback": record.feedback, "feedback_time": record.feedback_time,
                    "role": record.role, "content": record.content
                })

        return result


def delete_chat_session(session_id: str) -> bool:
    with SessionLocal() as session:
        session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if session_id is None:
            return False

        session.query(ChatMessageTable).where(ChatMessageTable.chat_id == session_id[0]).delete()
        session.query(ChatSessionTable).where(ChatSessionTable.id == session_id[0]).delete()
        session.commit()

    return True


def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if id is None:
            return False

        record = session.query(ChatMessageTable).filter(ChatMessageTable.id == message_id,
                                                        ChatMessageTable.chat_id == id[0]).first()
        if record is not None:
            record.feedback = feedback
            record.feedback_time = datetime.now()
            session.commit()

        return True


def list_chat(user_name: str) -> Optional[List[Any]]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records: Optional[List[ChatSessionTable]] = session.query(
                ChatSessionTable.user_id,
                ChatSessionTable.session_id,
                ChatSessionTable.title,
                ChatSessionTable.start_time).filter(ChatSessionTable.user_id == user_id[0]).all()
            if chat_records:
                return [ChatSession(user_id=x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time)
                        for x in chat_records]
            else:
                return []
        else:
            return []


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        message_recod = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if message_recod:
            message_recod = ChatMessageTable(
                chat_id=message_recod[0],
                role=role,
                content=content
            )
            session.add(message_recod)
            session.commit()

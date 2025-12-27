import os

os.environ["OPENAI_API_KEY"] = "sk-2acc31282cde4b149c8e5636d3394533"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_MODEL"] = "qwen-max"

from agents import set_default_openai_api, set_tracing_disabled

import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.extensions.memory import AdvancedSQLiteSession

from agents.mcp import MCPServerSse
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


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


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
) -> bool:
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

        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=""
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str):
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id,  # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )

    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    ) as mcp_server:
        stock_agent = Agent(
            name="stock_agent",
            instructions="You are a stock agent, good at asking stock-related questions, and tell me who you are when answering questions.",
            mcp_servers=[mcp_server],
            tool_use_behavior="run_llm_again",
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            )
        )

        chitchat_agent = Agent(
            name="chitchat_agent",
            instructions="You are a small talk agent, good at small talk, and tell me who you are when you answer questions。",
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            )
        )

        triage_agent = Agent(
            name="triage_agent",
            instructions="Handoff to the appropriate agent based on the language of the request.",
            handoffs=[chitchat_agent, stock_agent],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            )
        )

        result = Runner.run_streamed(triage_agent, input=content, session=session)

        assistant_message = ""
        current_tool_name = ""
        async for event in result.stream_events():
            # run llm again 的回答： 基础tool的结果继续回答
            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                            ResponseTextDeltaEvent):
                yield event.data.delta
                assistant_message += event.data.delta

    append_message2db(session_id, "assistant", assistant_message)


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        message_record = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if message_record:
            message_record = ChatMessageTable(
                chat_id=message_record[0],
                role=role,
                content=content
            )
            session.add(message_record)
            session.commit()
        return True


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

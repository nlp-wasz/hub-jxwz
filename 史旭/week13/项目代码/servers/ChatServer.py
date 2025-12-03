# 聊天 服务
import uuid
from typing import List

import httpx
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_api, AsyncOpenAI, OpenAIChatCompletionsModel, \
    ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from agents.mcp import MCPServerSse, ToolFilterStatic

from model_type.RequestResponse import ChatRequest, ChatSessionResponse, ChatMessageResponse
from model_type.SqliteOrm import SessionLocal, ChatSessionTable, ChatMessageTable


# 生成 session_id
def generator_session_id():
    # 通过 uuid 生成
    session_id = uuid.uuid4().hex

    # 判断数据库中是否存在
    with SessionLocal() as session:
        sel_all = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).all()
        if len(sel_all) > 0:
            raise Exception("session_id已存在")
        return session_id


# 判断 ChatSessionTable 表中是否已经存在 session_id（不存在则初始化）
async def session_init(
        prompt: str, user_name: str, session_id: str, message_role: str
):
    with SessionLocal() as session:
        try:
            sel_one = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id).first()
            if sel_one is None:
                # 初始化 ChatSessionTable 和 ChatMessageTable 表
                chatSession = ChatSessionTable(
                    session_id=session_id,
                    user_id=user_name,
                    chat_title=prompt,
                    chat_feedback=""
                )
                session.add(chatSession)

                # 关键：flush 让数据库生成 chat_id 并回填
                session.flush()

                chatMessage = ChatMessageTable(
                    chat_id=chatSession.chat_id,
                    user_id=user_name,
                    session_id=session_id,
                    message_role=message_role,
                    message_content=prompt,
                    message_feedback=""
                )
                session.add(chatMessage)

                session.commit()
            else:
                # 存在 session_id，因此只需要将后续信息添加到 ChatMessageTable 表
                chatMessage = ChatMessageTable(
                    chat_id=sel_one.chat_id,
                    user_id=user_name,
                    session_id=session_id,
                    message_role=message_role,
                    message_content=prompt,
                    message_feedback=""
                )
                session.add(chatMessage)
                session.commit()

            return True
        except Exception as e:
            session.rollback()
            return False


# 聊天方法
async def chat(chat_request: ChatRequest):
    # 将 用户请求的问题 进行缓存
    prompt: str = chat_request.prompt
    user_name: str = chat_request.user_name
    session_id: str = chat_request.session_id
    select_tools: List[str] = chat_request.select_tools  # 可以使用的 MCP 工具
    is_session_init = await session_init(prompt=prompt, user_name=user_name, session_id=session_id,
                                         message_role="user")
    if not is_session_init:
        yield "数据缓存失败，请重新发送信息！"
        return
    else:
        # 配置 Agent 环境
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")

        openai_client = AsyncOpenAI(
            api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model = OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=openai_client
        )

        # MCP SSE 连接
        print(f"select_tools:{select_tools}")
        mcp_sse = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://127.0.0.1:8002/sse"},
            cache_tools_list=False,
            client_session_timeout_seconds=20,
            tool_filter=ToolFilterStatic(allowed_tool_names=select_tools) if select_tools else None
        )

        try:
            async with mcp_sse:
                agent = Agent(
                    name="Chat Agent",
                    instructions="你是一个专业的聊天assistant",
                    model=model,
                    mcp_servers=[mcp_sse] if select_tools else [],
                    # model_settings=ModelSettings(parallel_tool_calls=False),  # 调用多个工具，关闭，默认只调用一个
                    tool_use_behavior="run_llm_again"
                )

                # agent 聊天信息缓存
                session = AdvancedSQLiteSession(
                    session_id=session_id,
                    db_path="./data/session.db",
                    create_tables=True
                )

                agent_res = Runner.run_streamed(agent, input=chat_request.prompt, session=session)
                # agent_res = Runner.run_streamed(agent, input=chat_request.prompt)

                # 记录 LLM 调用工具的结果（工具名称 和 工具参数 展示）
                func_res = ""

                # 记录 LLM生成的 回答
                llm_res = ""
                async for event in agent_res.stream_events():
                    if event.type == "raw_response_event":
                        if isinstance(event.data, ResponseOutputItemDoneEvent) and isinstance(event.data.item,
                                                                                              ResponseFunctionToolCall):
                            # 返回 工具调用信息
                            yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n```\n"
                            func_res += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n"

                        if isinstance(event.data, ResponseTextDeltaEvent):
                            yield event.data.delta

                            llm_res += event.data.delta

                # 工具 和 LLM 返回的结果 合并
                llm_mcp_res = func_res + llm_res

                # 保存大模型生成的 信息
                is_session_init = await session_init(prompt=llm_mcp_res, user_name=user_name, session_id=session_id,
                                                     message_role="assistant")
                if not is_session_init:
                    yield "数据缓存失败，请重新发送信息！"
                    return

        except Exception as e:
            yield "查询失败，请重新输入要查询的信息！"
            return


# 获取 缓存信息 列表（ChatSessionTable 表）
def get_chat_list(user_name):
    # 根据 用户名 查询 ChatSessionTable 表（user_id 字段）
    try:
        with SessionLocal() as session:
            sel_session = session.query(ChatSessionTable).filter(ChatSessionTable.user_id == user_name).all()

            if len(sel_session) > 0:
                # 返回 查询到的结果
                return [ChatSessionResponse(
                    chat_id=chatSession.chat_id,
                    session_id=chatSession.session_id,
                    user_id=chatSession.user_id,
                    chat_title=chatSession.chat_title,
                    chat_feedback=chatSession.chat_feedback,
                    feedback_at=chatSession.feedback_at.strftime("%Y-%m-%d %H:%M:%S"),
                    created_at=chatSession.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    update_at=chatSession.update_at.strftime("%Y-%m-%d %H:%M:%S"),
                ) for chatSession in sel_session]
            else:
                return []
    except Exception as e:
        return []


# 获取 缓存信息 列表（ChatMessageTable 表）
def get_message_list(session_id):
    # 根据 session_id 查询 ChatMessageTable 表
    try:
        with SessionLocal() as session:
            sel_message = session.query(ChatMessageTable).filter(ChatMessageTable.session_id == session_id).all()

            if len(sel_message) > 0:
                # 返回 查询到的结果
                return [ChatMessageResponse(
                    message_id=chatMessage.message_id,
                    chat_id=chatMessage.chat_id,
                    user_id=chatMessage.user_id,
                    session_id=chatMessage.session_id,

                    message_role=chatMessage.message_role,
                    message_content=chatMessage.message_content,
                    message_feedback=chatMessage.message_feedback,
                    feedback_at=chatMessage.feedback_at.strftime("%Y-%m-%d %H:%M:%S"),
                    created_at=chatMessage.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    update_at=chatMessage.update_at.strftime("%Y-%m-%d %H:%M:%S"),
                ) for chatMessage in sel_message]
            else:
                return []
    except Exception as e:
        return []


# 根据 session_id 删除缓存信息
def delete_chat_message(session_id):
    # 根据 session_id 查询 ChatMessageTable 表
    try:
        with SessionLocal() as session:
            session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).delete()
            session.query(ChatMessageTable).filter(ChatMessageTable.session_id == session_id).delete()
            session.commit()

            return True
    except Exception as e:
        session.rollback()
        return False

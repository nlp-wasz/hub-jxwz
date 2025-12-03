import os
import random
import string
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

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


def get_init_message(
        agent_type: str,
) -> List[Dict[Any, Any]]:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinjia2")

    if agent_type == "analysis":
        task_description = """
1. 专注于数据分析和逻辑推理任务。
2. 处理股票分析、数据BI、代码生成等结构化任务。
3. 必须使用专业、严谨的术语，确保分析结果的准确性。
4. 基于数据和事实进行分析，提供结构化的结论。
5. 当用户需求超出分析范围时，明确说明并建议切换到闲聊模式。
        """
    else:  # casual
        task_description = """
1. 专注于日常对话、情感交流及非结构化闲聊内容。
2. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
3. 避免过于专业或生硬的术语，除非用户明确要求。
4. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
5. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
6. 当用户需要数据分析或专业信息时，建议切换到分析模式。
        """

    system_prompt = template.render(
        agent_name="小呆助手" if agent_type == "casual" else "小呆分析师",
        task_description=task_description,
        current_datetime=datetime.now(),
    )
    return system_prompt


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        agent_type: str = "casual",
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

        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_init_message(agent_type)
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


def should_switch_to_analysis(content: str) -> bool:
    """
    判断是否应该从闲聊agent切换到分析agent
    """
    analysis_keywords = [
        "分析", "数据", "股票", "代码", "图表", "统计", "趋势", "预测",
        "计算", "比较", "评估", "模型", "算法", "查询", "报表", "K线",
        "市盈率", "财报", "投资", "收益", "风险", "涨跌幅", "成交量"
    ]
    
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in analysis_keywords)


def should_switch_to_casual(content: str) -> bool:
    """
    判断是否应该从分析agent切换到闲聊agent
    """
    casual_keywords = [
        "你好", "谢谢", "再见", "天气", "心情", "感觉", "喜欢", "讨厌",
        "故事", "笑话", "聊天", "闲聊", "日常", "生活", "情感", "兴趣",
        "爱好", "电影", "音乐", "美食", "旅游", "周末", "假期", "放松"
    ]
    
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in casual_keywords)


def detect_agent_type(content: str, current_agent_type: str) -> str:
    """
    根据用户输入和当前agent类型，决定应该使用哪种agent
    """
    if current_agent_type == "casual":
        return "analysis" if should_switch_to_analysis(content) else "casual"
    else:  # analysis
        return "casual" if should_switch_to_casual(content) else "analysis"


async def chat(user_name:str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    current_agent_type = "casual"  # 默认使用闲聊agent
    
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, current_agent_type)
            else:
                # 获取最近的系统消息，确定当前使用的agent类型
                last_system_message = session.query(ChatMessageTable).filter(
                    ChatMessageTable.chat_id == record.id,
                    ChatMessageTable.role == "system"
                ).order_by(ChatMessageTable.create_time.desc()).first()
                
                if last_system_message and "分析师" in last_system_message.content:
                    current_agent_type = "analysis"
    
    # 检测是否需要切换agent
    new_agent_type = detect_agent_type(content, current_agent_type)
    
    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)
    
    # 如果agent类型发生变化，记录切换事件
    if new_agent_type != current_agent_type:
        switch_message = f"[系统] 已从{'闲聊' if current_agent_type == 'casual' else '分析'}模式切换到{'分析' if new_agent_type == 'analysis' else '闲聊'}模式"
        append_message2db(session_id, "system", switch_message)
        yield switch_message + "\n\n"
        current_agent_type = new_agent_type
    
    # 获取system message，需要传给大模型，并不能给用户展示
    instructions = get_init_message(current_agent_type)

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
        session_id=session_id, # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 根据agent类型决定是否使用工具
    use_tools = (current_agent_type == "analysis" and tools and len(tools) > 0)
    
    # 如果没有选择工具或当前是闲聊agent，直接调用大模型回答
    if not use_tools:
        agent = Agent(
            name="Casual Assistant" if current_agent_type == "casual" else "Analysis Assistant",
            instructions=instructions,
            # mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            ),
            # tool_use_behavior="stop_on_first_tool",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        result = Runner.run_streamed(agent, input=content, session=session) # 流式调用大模型

        assistant_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent): # 如果式大模型的回答
                    if event.data.delta:
                        yield f"{event.data.delta}" # sse 不断发给前端
                        assistant_message += event.data.delta

        # 这一条大模型回答，存储对话
        append_message2db(session_id, "assistant", assistant_message)

    # 需要调用mcp 服务进行回答
    else:
        async with mcp_server:
            # 哪些工具直接展示结果
            need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
            if set(need_viz_tools) & set(tools):
                tool_use_behavior = "stop_on_first_tool" # 调用了tool，得到结果，就展示结果
            else:
                tool_use_behavior = "run_llm_again" # 调用了tool，得到结果，继续用大模型的总结结果

            agent = Agent(
                name="Analysis Assistant",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=os.environ["OPENAI_MODEL"],
                    openai_client=external_client,
                ),
                tool_use_behavior=tool_use_behavior,
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            result = Runner.run_streamed(agent, input=content, session=session)

            assistant_message = ""
            current_tool_name = ""
            async for event in result.stream_events():
                # if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output" and current_tool_name not in need_viz_tools:
                #     yield event.item.raw_item["output"]
                #     assistant_message += event.item.raw_item["output"]

                # tool_output
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name

                        # 工具名字、工具参数
                        yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                        assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                # run llm again 的回答： 基础tool的结果继续回答
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
                    assistant_message += event.data.delta


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
                return [ChatSession(user_id = x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time) for x in chat_records]
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

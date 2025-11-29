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
from agent.agent_with_handoffs import create_agents_with_handoffs


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
        task: str,
) -> List[Dict[Any, Any]]:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinjia2")

    if task == "股票分析":
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。

## 🚨 关键：调用K线工具的规则（必须严格遵守）

当用户要求查看股票走势图时，你必须使用以下工具之一：
- get_day_line: 日K线
- get_week_line: 周K线  
- get_month_line: 月K线

**🚨 极其重要 - 必须提供所有参数**：

尽管工具定义中 startDate 和 endDate 标记为"非必填"，但你必须始终提供这些参数，否则前端无法绘制图表。

必须提供的参数：
1. **code**: 股票代码（必填）
2. **startDate**: 开始日期，格式为 YYYY-MM-DD（必填！）
   - 如果用户未指定，使用今天往前推3个月的日期
   - 例如：今天是2024-11-27，则使用 "2024-08-27"
3. **endDate**: 结束日期，格式为 YYYY-MM-DD（必填！）
   - 如果用户未指定，使用今天的日期
   - 例如：今天是2024-11-27，则使用 "2024-11-27"
4. **type**: 复权类型（可选，默认0）
   - 0=不复权（默认）
   - 1=前复权
   - 2=后复权

**正确示例**：
get_month_line:{"code":"sh601169","startDate":"2024-08-27","endDate":"2024-11-27","type":0}

**错误示例（绝对不要这样做）**：
get_month_line:{"code":"sh601169"}  ❌ 缺少日期参数，会导致前端报错
"""
    elif task == "数据BI":
        task_description = """
1. 帮助用户理解他们的数据结构、商业指标和关键绩效指标 (KPI)。
2. 用户的请求通常是数据查询、指标定义或图表生成建议。
3. **关键约束：你的输出必须是可执行的代码块 (如 SQL 或 Python)，或者清晰的逻辑步骤，用于解决用户的数据问题。**
4. 严格遵守数据分析的逻辑严谨性，确保每一个结论都有数据支撑。
5. 当被要求提供可视化建议时，请推荐最合适的图表类型（如：时间序列用折线图，分类对比用柱状图）。"""
    else:
        task_description = """
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
5. 关键词：友好、轻松、富有同理心。

## 🚨 关键：调用K线工具的规则（必须严格遵守）

当用户要求查看股票走势图时，你必须使用以下工具之一：
- get_day_line: 日K线
- get_week_line: 周K线  
- get_month_line: 月K线

**🚨 极其重要 - 必须提供所有参数**：

尽管工具定义中 startDate 和 endDate 标记为"非必填"，但你必须始终提供这些参数，否则前端无法绘制图表。

必须提供的参数：
1. **code**: 股票代码（必填）
2. **startDate**: 开始日期，格式为 YYYY-MM-DD（必填！）
   - 如果用户未指定，使用今天往前推3个月的日期
   - 例如：今天是2024-11-27，则使用 "2024-08-27"
3. **endDate**: 结束日期，格式为 YYYY-MM-DD（必填！）
   - 如果用户未指定，使用今天的日期
   - 例如：今天是2024-11-27，则使用 "2024-11-27"
4. **type**: 复权类型（可选，默认0）

**正确示例**：
get_month_line:{"code":"sh601169","startDate":"2024-08-27","endDate":"2024-11-27","type":0}

**错误示例（绝对不要这样做）**：
get_month_line:{"code":"sh601169"}  ❌ 缺少日期参数，会导致前端报错
        """

    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(),
    )
    return system_prompt


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

        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_init_message(task)
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


async def chat(user_name:str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # 获取system message，需要传给大模型，并不能给用户展示
    instructions = get_init_message(task)

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
        client_session_timeout_seconds=120,
    )

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id, # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 如果没有选择工具，默认直接调用大模型回答
    if not tools or len(tools) == 0:
        agent = Agent(
            name="Assistant",
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

            # 🔍 调试信息：原有 chat() 函数
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG - ORIGINAL CHAT] 配置信息:")
            print(f"  - 用户输入: {content}")
            print(f"  - 选择的工具: {tools}")
            print(f"  - task类型: {task}")
            print(f"  - tool_use_behavior: {tool_use_behavior}")
            print(f"  - instructions前100字符: {instructions[:100]}...")
            print(f"{'='*60}\n")

            agent = Agent(
                name="Assistant",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=os.environ["OPENAI_MODEL"],
                    openai_client=external_client,
                ),
                tool_use_behavior=tool_use_behavior,
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            # 🔍 调试信息：Agent配置
            print(f"🔍 [DEBUG - ORIGINAL CHAT] Agent配置:")
            print(f"  - name: {agent.name}")
            print(f"  - tool_use_behavior: {agent.tool_use_behavior}")
            print(f"  - mcp_servers数量: {len(agent.mcp_servers)}")
            print(f"\n")

            result = Runner.run_streamed(agent, input=content, session=session)

            assistant_message = ""
            current_tool_name = ""
            event_count = 0
            
            async for event in result.stream_events():
                event_count += 1
                print(f"🔍 [DEBUG - ORIGINAL CHAT] Event #{event_count}: type={event.type}")
                
                # if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output" and current_tool_name not in need_viz_tools:
                #     yield event.item.raw_item["output"]
                #     assistant_message += event.item.raw_item["output"]

                # tool_output
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name

                        # 🔍 调试信息：工具调用
                        print(f"\n{'='*60}")
                        print(f"🔍 [DEBUG - ORIGINAL CHAT] 工具调用:")
                        print(f"  - 工具名: {event.data.item.name}")
                        print(f"  - 参数: {event.data.item.arguments}")
                        print(f"{'='*60}\n")

                        # 工具名字、工具参数
                        yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                        assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                # run llm again 的回答： 基础tool的结果继续回答
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
                    assistant_message += event.data.delta

            # 🔍 调试信息：执行完成
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG - ORIGINAL CHAT] 执行完成:")
            print(f"  - 总事件数: {event_count}")
            print(f"  - 最后调用的工具: {current_tool_name if current_tool_name else '无'}")
            print(f"{'='*60}\n")

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


async def chat_with_handoffs(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    """
    Chat function with agent handoffs support.
    Automatically routes between chat_agent and stock_agent based on user input.
    
    Args:
        user_name: Username
        session_id: Chat session ID
        task: Task type (used for context, but handoffs determine routing)
        content: User message content
        tools: List of MCP tools to enable
    """
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
        client_session_timeout_seconds=120,
    )

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 无论是否选择工具，都使用 handoffs
    # 即使没有选择工具，Agent 之间也可以转接
    if not tools or len(tools) == 0:
        # 无工具情况下，使用默认行为，但仍然启用 handoffs
        tool_use_behavior = "run_llm_again"
        
        print(f"\n{'='*60}")
        print(f"🔍 [DEBUG - NO TOOLS] chat_with_handoffs 配置信息:")
        print(f"  - 用户输入: {content}")
        print(f"  - 选择的工具: 无")
        print(f"  - task类型: {task}")
        print(f"  - tool_use_behavior: {tool_use_behavior}")
        print(f"{'='*60}\n")
        
        # 即使没有工具，也创建带 handoffs 的 Agent
        # ChatAgent 没有 MCP 工具，StockAgent 有 MCP 工具
        chat_agent, stock_agent = create_agents_with_handoffs(
            external_client=external_client,
            mcp_server=mcp_server,  # 传递 MCP 服务器给 StockAgent
            tools=None,
            tool_use_behavior=tool_use_behavior
        )

        # 根据task类型选择初始agent（默认总是 ChatAgent）
        initial_agent = chat_agent
        
        print(f"🔍 [DEBUG - NO TOOLS] 初始 Agent: {initial_agent.name}\n")

        # 使用 MCP 服务器（即使没有选择工具，StockAgent 仍需要访问所有工具）
        async with mcp_server:
            result = Runner.run_streamed(initial_agent, input=content, session=session)

            assistant_message = ""
            current_agent = initial_agent.name
            current_tool_name = ""
            event_count = 0
            
            async for event in result.stream_events():
                event_count += 1
                print(f"🔍 [DEBUG - NO TOOLS] Event #{event_count}: type={event.type}")
                
                # 检测 Agent 切换
                if event.type == "agent_updated_stream_event":
                    if hasattr(event, 'new_agent') and hasattr(event.new_agent, 'name'):
                        new_agent_name = event.new_agent.name
                        print(f"🔍 [DEBUG - NO TOOLS] Agent 切换: {current_agent} → {new_agent_name}")
                        if new_agent_name != current_agent:
                            handoff_msg = f"\n\n🔄 **Agent 转接**: {current_agent} → {new_agent_name}\n\n"
                            print(f"🔍 [DEBUG - NO TOOLS] {handoff_msg.strip()}")
                            yield handoff_msg
                            assistant_message += handoff_msg
                            current_agent = new_agent_name
                
                # 处理工具调用
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name
                        print(f"🔍 [DEBUG - NO TOOLS] 工具调用: {event.data.item.name}")
                        # 过滤 handoff 工具
                        if not event.data.item.name.startswith("transfer_to_"):
                            yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                            assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                
                # 处理文本输出
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        yield event.data.delta
                        assistant_message += event.data.delta
            
            print(f"🔍 [DEBUG - NO TOOLS] 执行完成，总事件数: {event_count}\n")

        # 存储对话
        append_message2db(session_id, "assistant", assistant_message)

    # 需要调用mcp 服务进行回答
    else:
        async with mcp_server:
            # 哪些工具直接展示结果
            need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
            if set(need_viz_tools) & set(tools):
                tool_use_behavior = "stop_on_first_tool"  # 调用了tool，得到结果，就展示结果
            else:
                tool_use_behavior = "run_llm_again"  # 调用了tool，得到结果，继续用大模型的总结结果

            # 🔍 调试信息1：打印配置
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG] chat_with_handoffs 配置信息:")
            print(f"  - 用户输入: {content}")
            print(f"  - 选择的工具: {tools}")
            print(f"  - task类型: {task}")
            print(f"  - tool_use_behavior: {tool_use_behavior}")
            print(f"  - 是否有MCP服务器: {mcp_server is not None}")
            print(f"{'='*60}\n")

            chat_agent, stock_agent = create_agents_with_handoffs(
                external_client=external_client,
                mcp_server=mcp_server,
                tools=tools if tools and len(tools) > 0 else None,
                tool_use_behavior=tool_use_behavior
            )

            # 🔍 调试信息2：打印Agent配置
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG] Agent 配置信息:")
            print(f"  - ChatAgent:")
            print(f"    - name: {chat_agent.name}")
            print(f"    - tool_use_behavior: {chat_agent.tool_use_behavior}")
            print(f"    - mcp_servers数量: {len(chat_agent.mcp_servers) if hasattr(chat_agent, 'mcp_servers') else 0}")
            print(f"  - StockAgent:")
            print(f"    - name: {stock_agent.name}")
            print(f"    - tool_use_behavior: {stock_agent.tool_use_behavior}")
            print(f"    - mcp_servers数量: {len(stock_agent.mcp_servers) if hasattr(stock_agent, 'mcp_servers') else 0}")
            print(f"{'='*60}\n")

            # 根据task类型选择初始agent
            if task == "股票分析":
                initial_agent = stock_agent
            else:
                initial_agent = chat_agent

            # 🔍 调试信息3：打印初始Agent
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG] 初始 Agent: {initial_agent.name}")
            print(f"{'='*60}\n")

            # 运行agent流
            result = Runner.run_streamed(initial_agent, input=content, session=session)

            assistant_message = ""
            current_tool_name = ""
            current_agent = initial_agent.name
            event_count = 0

            async for event in result.stream_events():
                event_count += 1
                
                # 🔍 调试信息4：打印所有事件
                print(f"🔍 [DEBUG] Event #{event_count}: type={event.type}")
                
                # 检测 Agent 切换事件
                if event.type == "agent_updated_stream_event":
                    if hasattr(event, 'new_agent') and hasattr(event.new_agent, 'name'):
                        new_agent_name = event.new_agent.name
                        print(f"🔍 [DEBUG] Agent 切换: {current_agent} → {new_agent_name}")
                        if new_agent_name != current_agent:
                            handoff_msg = f"\n\n🔄 **Agent 转接**: {current_agent} → {new_agent_name}\n\n"
                            print(f"🔍 [DEBUG] {handoff_msg.strip()}")
                            yield handoff_msg
                            assistant_message += handoff_msg
                            current_agent = new_agent_name
                
                # 处理工具调用
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name
                        
                        # 🔍 调试信息5：打印工具调用详情
                        print(f"\n{'='*60}")
                        print(f"🔍 [DEBUG] 工具调用:")
                        print(f"  - 工具名: {event.data.item.name}")
                        print(f"  - 参数: {event.data.item.arguments}")
                        print(f"{'='*60}\n")

                        # 过滤掉 handoff 工具调用的输出（这些是内部转接，不需要显示给用户）
                        if not event.data.item.name.startswith("transfer_to_"):
                            # 工具名字、工具参数
                            yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                            assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                # run llm again 的回答： 基础tool的结果继续回答
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        print(f"🔍 [DEBUG] 文本输出: {event.data.delta[:50]}...")
                    yield event.data.delta
                    assistant_message += event.data.delta

            # 🔍 调试信息6：总结
            print(f"\n{'='*60}")
            print(f"🔍 [DEBUG] 执行完成:")
            print(f"  - 总事件数: {event_count}")
            print(f"  - 最后调用的工具: {current_tool_name if current_tool_name else '无'}")
            print(f"  - 响应长度: {len(assistant_message)} 字符")
            print(f"{'='*60}\n")

            # 存储对话
            append_message2db(session_id, "assistant", assistant_message)
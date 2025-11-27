import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, handoff
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


# ============================================================
# Agent Instructions 定义
# ============================================================

TRIAGE_AGENT_INSTRUCTIONS = """
你是一个智能路由助手，你的唯一职责是分析用户的问题，并将其转交给合适的专业Agent处理。

## 判断规则：

### 转交给 **股票分析Agent** 的情况：
- 用户询问任何股票、证券、基金相关问题
- 涉及股价、K线、技术指标（MACD, KDJ, RSI等）
- 询问公司财报、市盈率(P/E)、市净率(P/B)、EPS等财务指标
- 讨论投资策略、选股、仓位管理
- 询问大盘走势、板块行情
- 提及具体股票代码或公司名称的投资相关问题

### 转交给 **闲聊Agent** 的情况：
- 日常问候、寒暄（你好、早上好、在吗等）
- 闲聊话题（天气、心情、生活琐事）
- 通用知识问答（非金融专业领域）
- 笑话、故事、娱乐内容
- 用户表达情感需要倾听和陪伴

## 重要提示：
- 不要自己回答问题，你的职责仅是路由
- 分析完用户意图后，立即调用对应的handoff函数
- 如果不确定，默认转交给闲聊Agent
"""

CHAT_AGENT_INSTRUCTIONS = """
你是「小呆助手」，一个友好、温暖、富有同理心的闲聊伙伴。

## 你的性格特点：
- 🌟 **友好热情**：用轻松愉快的语气与用户交流
- 💭 **善于倾听**：认真理解用户的表达，给予适当回应
- 😊 **富有幽默感**：适时加入一些轻松的元素
- 🤗 **有同理心**：当用户分享情绪时，先共情再回应

## 对话原则：
1. 保持回答简洁自然，不要过于冗长
2. 避免使用生硬的专业术语
3. 适当使用表情符号增加亲和力
4. 当用户提到股票/投资相关话题时，建议他们可以问我关于股票的专业问题

当前时间：{current_datetime}
"""

STOCK_AGENT_INSTRUCTIONS = """
你是「小呆助手」的股票分析专家模块，专注于提供专业、严谨的金融分析服务。

## 专业领域：
- 全球主要股票市场分析（NYSE, NASDAQ, SSE, SZSE, HKEX）
- 技术分析：K线形态、均线系统、MACD、KDJ、RSI、布林带等
- 基本面分析：财务报表解读、估值模型、行业对比
- 量化指标：P/E, P/B, ROE, EPS, Beta, 夏普比率等

## 分析框架：
1. **公司概览**：主营业务、行业地位、竞争优势（护城河）
2. **财务健康度**：盈利能力、偿债能力、运营效率
3. **估值分析**：相对估值（同业对比）、绝对估值（DCF等）
4. **风险与机遇**：宏观风险、行业风险、公司特有风险

## ⚠️ 重要声明：
- 所有分析仅基于公开市场数据
- 分析结果仅供参考，**不构成投资建议**
- 投资有风险，入市需谨慎
- 请用户根据自身风险承受能力做出决策

## 当用户想闲聊时：
如果用户开始闲聊或问非股票问题，友好地告诉他们可以随时聊其他话题。

当前时间：{current_datetime}
"""


def get_formatted_instructions(template: str) -> str:
    """格式化指令模板，填入当前时间等动态信息"""
    return template.format(current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================
# Agent 工厂函数
# ============================================================

def create_agents(external_client: AsyncOpenAI, mcp_server: Optional[MCPServerSse] = None):
    """
    创建多Agent系统，包含Triage、Chat、Stock三个Agent
    通过handoff机制实现协作
    """
    
    model = OpenAIChatCompletionsModel(
        model=os.environ["OPENAI_MODEL"],
        openai_client=external_client,
    )
    
    # 1. 创建闲聊Agent
    chat_agent = Agent(
        name="闲聊助手",
        instructions=get_formatted_instructions(CHAT_AGENT_INSTRUCTIONS),
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    
    # 2. 创建股票分析Agent（可以配置MCP工具）
    stock_agent_config = {
        "name": "股票分析专家",
        "instructions": get_formatted_instructions(STOCK_AGENT_INSTRUCTIONS),
        "model": model,
        "model_settings": ModelSettings(parallel_tool_calls=False),
    }
    
    if mcp_server:
        stock_agent_config["mcp_servers"] = [mcp_server]
        stock_agent_config["tool_use_behavior"] = "run_llm_again"
    
    stock_agent = Agent(**stock_agent_config)
    
    # 3. 创建Triage路由Agent，配置handoff
    triage_agent = Agent(
        name="智能路由",
        instructions=TRIAGE_AGENT_INSTRUCTIONS,
        model=model,
        handoffs=[
            handoff(
                agent=chat_agent,
                tool_name_override="transfer_to_chat_agent",
                tool_description_override="将用户转交给闲聊助手处理日常对话、问候、情感交流等非专业问题"
            ),
            handoff(
                agent=stock_agent,
                tool_name_override="transfer_to_stock_agent", 
                tool_description_override="将用户转交给股票分析专家处理股票、证券、投资、财务分析等金融专业问题"
            ),
        ],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    
    return triage_agent, chat_agent, stock_agent


# ============================================================
# 保留原有的辅助函数
# ============================================================

def get_init_message(task: str) -> str:
    """保留原有函数，用于向后兼容"""
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinjia2")

    if task == "股票分析":
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
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

        # 使用新的多Agent系统，不再需要存储单一的system message
        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content="Multi-Agent System Initialized"
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


# ============================================================
# 核心对话函数 - 使用Handoff多Agent架构
# ============================================================

async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    """
    多Agent对话入口
    通过Triage Agent自动判断用户意图，handoff到对应的专业Agent
    """
    
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # 初始化OpenAI客户端
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # openai-agent支持的session存储
    agent_session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # MCP工具配置（仅用于股票Agent）
    mcp_server = None
    if tools and len(tools) > 0:
        tool_mcp_tools_filter = ToolFilterStatic(allowed_tool_names=tools)
        mcp_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            tool_filter=tool_mcp_tools_filter,
            client_session_timeout_seconds=20,
        )

    # 需要可视化的工具列表
    need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]

    assistant_message = ""

    # 根据是否有MCP工具决定执行方式
    if mcp_server:
        async with mcp_server:
            # 创建多Agent系统
            triage_agent, chat_agent, stock_agent = create_agents(external_client, mcp_server)
            
            # 使用Triage Agent作为入口运行
            result = Runner.run_streamed(
                triage_agent, 
                input=content, 
                session=agent_session
            )

            current_tool_name = ""
            current_agent_name = ""
            
            async for event in result.stream_events():
                # 捕获Agent切换事件
                if event.type == "agent_updated_stream_event":
                    new_agent_name = event.new_agent.name
                    if new_agent_name != current_agent_name:
                        current_agent_name = new_agent_name
                        # 可选：通知前端当前是哪个Agent在处理
                        agent_indicator = f"\n> 🤖 *{current_agent_name}* 正在为您服务...\n\n"
                        yield agent_indicator
                        assistant_message += agent_indicator

                # 处理工具调用输出
                if event.type == "raw_response_event" and hasattr(event, 'data'):
                    if isinstance(event.data, ResponseOutputItemDoneEvent):
                        if isinstance(event.data.item, ResponseFunctionToolCall):
                            tool_name = event.data.item.name
                            
                            # 跳过handoff相关的工具调用显示
                            if tool_name.startswith("transfer_to_"):
                                continue
                                
                            current_tool_name = tool_name
                            tool_output = f"\n```json\n{tool_name}: {event.data.item.arguments}\n```\n\n"
                            yield tool_output
                            assistant_message += tool_output

                    # 处理文本流式输出
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        if event.data.delta:
                            yield event.data.delta
                            assistant_message += event.data.delta

    else:
        # 无MCP工具的情况
        triage_agent, chat_agent, stock_agent = create_agents(external_client)
        
        result = Runner.run_streamed(
            triage_agent, 
            input=content, 
            session=agent_session
        )

        current_agent_name = ""
        
        async for event in result.stream_events():
            # 捕获Agent切换事件
            if event.type == "agent_updated_stream_event":
                new_agent_name = event.new_agent.name
                if new_agent_name != current_agent_name:
                    current_agent_name = new_agent_name
                    agent_indicator = f"\n> 🤖 *{current_agent_name}* 正在为您服务...\n\n"
                    yield agent_indicator
                    assistant_message += agent_indicator

            # 处理文本流式输出
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        yield event.data.delta
                        assistant_message += event.data.delta

    # 存储助手回复
    append_message2db(session_id, "assistant", assistant_message)


# ============================================================
# 保留原有的数据库操作函数（无修改）
# ============================================================

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
                return [ChatSession(user_id=x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time) for x in chat_records]
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

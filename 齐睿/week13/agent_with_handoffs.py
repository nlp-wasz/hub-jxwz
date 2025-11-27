"""
Agent with handoffs implementation for chat service.
Supports two agents: chat_agent (闲聊) and stock_agent (股票分析)
with handoff capability between them.
"""

import os
from typing import Optional, List, Union
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, StopAtTools, handoff
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI


def create_chat_agent(
    external_client: AsyncOpenAI,
    mcp_server: Optional[MCPServerSse] = None,
    tool_use_behavior: Union[str, StopAtTools, None] = None,
) -> Agent:
    """
    Create a chat agent for casual conversation.
    This agent handles general chat and can handoff to stock_agent when needed.
    """
    # 转换字符串为正确的类型
    if tool_use_behavior == "stop_on_first_tool":
        tool_use_behavior = StopAtTools()
    elif tool_use_behavior == "run_llm_again" or tool_use_behavior == "auto" or tool_use_behavior is None:
        # 使用 "run_llm_again" 作为默认行为
        tool_use_behavior = "run_llm_again"
    
    instructions = """
你是一个友好、轻松的对话助手(ChatAgent)。你的职责是：
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
5. 关键词：友好、轻松、富有同理心。

当用户询问关于股票、财务分析、市场数据等专业金融问题时，请转接给 StockAgent。
"""

    mcp_servers = [mcp_server] if mcp_server else []

    agent = Agent(
        name="ChatAgent",
        instructions=instructions,
        mcp_servers=mcp_servers,
        model=OpenAIChatCompletionsModel(
            model=os.environ.get("OPENAI_MODEL", "gpt-4"),
            openai_client=external_client,
        ),
        tool_use_behavior=tool_use_behavior,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    return agent


def create_stock_agent(
    external_client: AsyncOpenAI,
    mcp_server: Optional[MCPServerSse] = None,
    tool_use_behavior: Union[str, StopAtTools, None] = None,
) -> Agent:
    """
    Create a stock analysis agent.
    This agent handles stock analysis and financial queries.
    Can handoff to chat_agent for general conversation.
    """
    # 转换字符串为正确的类型
    if tool_use_behavior == "stop_on_first_tool":
        tool_use_behavior = StopAtTools()
    elif tool_use_behavior == "run_llm_again" or tool_use_behavior == "auto" or tool_use_behavior is None:
        # 使用 "run_llm_again" 作为默认行为
        tool_use_behavior = "run_llm_again"
    
    instructions = """
你是一个专业的股票分析助手。你的职责是：
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. 在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。
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

当用户询问与股票分析无关的话题（如天气、闲聊等）时，使用 transfer_to_chat 函数将对话转接给 ChatAgent。
"""

    mcp_servers = [mcp_server] if mcp_server else []

    agent = Agent(
        name="StockAgent",
        instructions=instructions,
        mcp_servers=mcp_servers,
        model=OpenAIChatCompletionsModel(
            model=os.environ.get("OPENAI_MODEL", "gpt-4"),
            openai_client=external_client,
        ),
        tool_use_behavior=tool_use_behavior,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    return agent


def create_agents_with_handoffs(
    external_client: AsyncOpenAI,
    mcp_server: Optional[MCPServerSse] = None,
    tools: Optional[List[str]] = None,
    tool_use_behavior: Union[str, StopAtTools, None] = None,
) -> tuple[Agent, Agent]:
    """
    Create both chat and stock agents with handoff capability.

    Args:
        external_client: AsyncOpenAI client
        mcp_server: Optional MCP server for tools
        tools: Optional list of tool names to filter
        tool_use_behavior: How to handle tool calls ("auto", "stop_on_first_tool", "run_llm_again")

    Returns:
        Tuple of (chat_agent, stock_agent)
    """
    # Filter MCP server tools if specified
    if tools and len(tools) > 0:
        tool_filter = ToolFilterStatic(allowed_tool_names=tools)
        if mcp_server:
            mcp_server.tool_filter = tool_filter

    # 创建 Agent
    # ChatAgent: No MCP tools, only handoff capability
    # StockAgent: Full MCP tools access
    chat_agent = create_chat_agent(external_client, mcp_server=None, tool_use_behavior=tool_use_behavior)
    stock_agent = create_stock_agent(external_client, mcp_server=mcp_server, tool_use_behavior=tool_use_behavior)

    # 根据官方文档，使用 handoff() 函数并直接赋值给 Agent
    # 这会自动注册 handoff 工具
    chat_agent.handoffs = [handoff(stock_agent)]
    stock_agent.handoffs = [handoff(chat_agent)]
    
    print(f"🔍 [DEBUG] Handoffs configured:")
    print(f"  - ChatAgent.handoffs: {chat_agent.handoffs}")
    print(f"  - StockAgent.handoffs: {stock_agent.handoffs}")

    return chat_agent, stock_agent

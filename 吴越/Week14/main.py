import os

os.environ["OPENAI_AI_KEY"]="sk-65b8e3a30263430f99da2cc286004704"
os.environ["OPENAI_BASE_URL"]="https://dashscope.aliyuncs.com/compatible-mode/v1"
from agents import Agent,Runner
from agents.mcp.server import MCPServerSse
from agents.mcp import MCPServer,ToolFilterStatic
from agents import set_default_openai_api, set_tracing_disabled
from tool_selector import Tool_Selector
from langchain_core.prompts import PromptTemplate

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def qa(query,topk):
    template = '''
       你是专业的AI计算助手，负责使用给定的工具集解决用户问题。你已经连接了特定的计算工具，这些工具是根据你的问题自动筛选出来的最相关工具。你可以直接使用这些工具进行计算。
        # 📋 工作流程
        1. **分析需求**：仔细理解用户的计算需求
        2. **选择工具**：从当前可用工具中选择最适合的一个
        3. **确认参数**：如有缺失参数，请用户提供
        4. **执行计算**：调用工具进行计算
        5. **解释结果**：清晰地展示并解释计算结果
       问题：{query}
       '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"])
    tools=Tool_Selector().get_similarity_tool(query,topk)
    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    async with MCPServerSse(
        name="MCPServerSse",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    ) as server:
        agent=Agent(
            model="Qwen3-235B-A22B-Thinking-2507",
            name="Assistant",
            instructions=prompt
        )

        result = Runner.run_streamed(agent, input=query)
        return result.final_output
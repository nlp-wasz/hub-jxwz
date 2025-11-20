# mcp 客户端，连接mcp服务端，并调用tools
import asyncio, json, os
from typing import Annotated

from fastmcp import Client
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled
from agents.mcp import MCPServerSse

# 环境配置
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
os.environ["OPENAI_API_KEY"] = "sk-04ab3d7290e243dda1badc5a1d5ac858"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# mcp http模式
async def mcp_http():
    client = Client("http://127.0.0.1:8000/mcp")
    async with client:
        res = await client.call_tool(name="get_today_daily_news")
        print(json.dumps(json.loads(res.content[0].text), indent=4, ensure_ascii=False))


# mcp sse模式（与 Agent 配合使用）
async def mcp_sse_agent(input: Annotated[str, "你是谁？"]):
    # mcp sse 连接
    mcp_sse_server = MCPServerSse(name="Mcp_sse 模式连接方式", params={"url": "http://127.0.0.1:8000/sse"})

    async with mcp_sse_server:
        # 创建 Agent
        mcp_sse_agent = Agent(
            name="可调用mcp工具的Agent",
            model="qwen-max",
            instructions="你是一个专业的assistant，并且可以使用mcp工具。",
            mcp_servers=[mcp_sse_server]
        )

        runner_res = await Runner.run(mcp_sse_agent, input)
        # print(runner_res.final_output)

        return runner_res.final_output

# if __name__ == '__main__':
#     # mcp http模式
#     # asyncio.run(mcp_http())
#
#     # mcp sse模式（与 Agent 配合使用）
#     # asyncio.run(mcp_sse_agent("北京的天气如何？"))
#     asyncio.run(mcp_sse_agent("每日新闻今天有什么新闻列表？"))

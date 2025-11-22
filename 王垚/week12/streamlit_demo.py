import os
import asyncio

import streamlit as st

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    SQLiteSession,
    ModelSettings,
)
from agents.mcp import MCPServer
from agents.mcp.server import MCPServerSse
from openai.types.responses import ResponseTextDeltaEvent
from agents import set_default_openai_api, set_tracing_disabled


# --- OpenAI-Agent 基础配置（复现示例，但在 my_job 项目中重写） ---
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


st.set_page_config(page_title="职能助手 (my_job)")
session = SQLiteSession("my_job_conversation")


# --- 侧边栏 ---
with st.sidebar:
    st.title("职能AI + 智能问答 (my_job)")

    api_key_default = os.environ.get("OPENAI_API_KEY", "")
    if "API_TOKEN" in st.session_state and st.session_state["API_TOKEN"]:
        key = st.session_state["API_TOKEN"]
        st.success("API Token 已配置", icon="✅")
    else:
        key = api_key_default

    key = st.text_input("输入 Token:", type="password", value=key)
    st.session_state["API_TOKEN"] = key

    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"], index=0)
    use_tool = st.checkbox("允许调用 MCP 工具", value=True)


# --- 初始化对话历史 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手 (my_job)，可以直接 AI 对话，也可以在需要时调用内部工具（新闻、工具、情感分析等）。"}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history() -> None:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手 (my_job)，可以直接 AI 对话，也可以在需要时调用内部工具（新闻、工具、情感分析等）。"}
    ]
    global session
    session = SQLiteSession("my_job_conversation")


st.sidebar.button("清空聊天", on_click=clear_chat_history)


# --- tool_filter 逻辑 ---
# 需求：
#   - 查询新闻的时候，只调用 news 的工具
#   - 调用工具的时候，只调用 tools 的工具
# 简单实现：基于当前用户输入 prompt 来做前缀/关键字判断。


def build_tool_filter(prompt: str):
    """根据当前用户输入构造一个 tool_filter 函数，传给 MCPServerSse。

    tool.name 是 fastmcp tool 名字；我们在 mcp_server_main.py 里给不同子 server
    加了前缀： news- / saying- / tool-，也可以根据名字包含关键字来筛选。
    """

    lower_prompt = prompt.lower()

    # MCPServerSse 在调用 tool_filter 时会传入 (tool, context)，这里第二个参数用不到，接收后忽略即可
    def _filter(tool, _context=None):
        name = getattr(tool, "name", "") or ""

        # 查询新闻：只允许 news server 相关的工具
        if "新闻" in prompt or "news" in lower_prompt:
            return name.startswith("news-")

        # 其它显式调用 "工具" 或常规业务：偏向 tools server
        if "工具" in prompt or "tool" in lower_prompt:
            return name.startswith("tool-")

        # 默认：放开全部（news + tools + saying），让 agent 自己选择
        return True

    return _filter


# --- 与模型交互的异步函数 ---


async def get_model_response(prompt: str, model_name: str, use_tool: bool):
    async with MCPServerSse(
        name="SSE Python Server (my_job)",
        params={
            "url": "http://localhost:8900/sse",  # 对应 my_job/mcp_server_main.py 中的 SSE 端口
        },
        cache_tools_list=False,
        tool_filter=build_tool_filter(prompt),  # 关键：在这里挂上 tool_filter
        client_session_timeout_seconds=20,
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=st.session_state.get("API_TOKEN", ""),
            base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )

        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="你是一个企业职能助手，可以在合适的时候调用 MCP 工具（新闻查询、工具服务、情感分析等）来帮助用户。",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                # 开启“允许调用 MCP 工具”时，强制要求本轮对话至少调用一次工具，方便调试和演示
                model_settings=ModelSettings(tool_choice="required"),
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="你是一个企业职能助手，只进行纯对话回答，不调用工具。",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
            )

        # 使用 openai-agent 的流式接口
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


# --- 主对话区域 ---

if len(st.session_state.get("API_TOKEN", "")) > 0:
    if prompt := st.chat_input("请输入你的问题或需求..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    response_generator = get_model_response(prompt, model_name, use_tool)

                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text

                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)
                except Exception as e:  # noqa: BLE001 - 简化示例
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message

            st.session_state.messages.append({"role": "assistant", "content": full_response})

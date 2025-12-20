import streamlit as st
from typing import Optional, Set, Dict, Any

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")

ASSISTANT_INSTRUCTIONS = """
你是企业职能助手，负责解答制度、业务以及资讯问题。
当需要天气、地址、货币等具体查询时，请调用 Tools 服务器提供的接口；
当用户想了解新闻或热点时，调用 News 服务器；
遇到需要进行情感或语气判断的文本，请使用情感分析工具。
直接回答聊天问题，并在无法调用工具时解释原因。
""".strip()

NEWS_TOOL_NAMES: Set[str] = {
    "get_today_daily_news",
    "get_douyin_hot_news",
    "get_github_hot_news",
    "get_toutiao_hot_news",
    "get_sports_news",
}

TOOLS_TOOL_NAMES: Set[str] = {
    "get_city_weather",
    "get_address_detail",
    "get_tel_info",
    "get_scenic_info",
    "get_flower_info",
    "get_rate_transform",
}

SENTIMENT_TOOL_NAMES: Set[str] = {"analyze_text_sentiment"}


def resolve_tool_filter(user_prompt: str) -> Optional[Dict[str, Any]]:
    normalized = user_prompt.lower()
    if "新闻" in user_prompt or "news" in normalized:
        return {
            "label": "仅使用新闻相关工具",
            "allowed_names": NEWS_TOOL_NAMES,
        }
    if "工具" in user_prompt or "tool" in normalized:
        return {
            "label": "仅使用Tools工具集",
            "allowed_names": TOOLS_TOOL_NAMES,
        }
    if (
        any(keyword in user_prompt for keyword in ("情感", "情绪", "心情"))
        or "sentiment" in normalized
        or "emotion" in normalized
    ):
        return {
            "label": "仅使用情感分析工具",
            "allowed_names": SENTIMENT_TOOL_NAMES,
        }
    return None


class MCPServerToolFilterAdapter:
    """Wraps an MCP server and exposes only a subset of tools."""

    def __init__(self, base_server: MCPServer, allowed_names: Set[str]):
        self._base_server = base_server
        self._allowed_names = set(allowed_names)

    async def list_tools(self):
        tools = await self._base_server.list_tools()
        if not self._allowed_names:
            return tools
        return [tool for tool in tools if tool.name in self._allowed_names]

    async def call_tool(self, name, *args, **kwargs):
        if self._allowed_names and name not in self._allowed_names:
            raise ValueError(f"Tool '{name}' is not allowed in the current context.")
        return await self._base_server.call_tool(name, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._base_server, item)


with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")

if "tool_filter_status" not in st.session_state:
    st.session_state["tool_filter_status"] = "未启用工具调用"

with st.sidebar:
    st.markdown("**工具调度策略**")
    st.caption(st.session_state["tool_filter_status"])

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)

async def get_model_response(prompt, model_name, use_tool, filter_info: Optional[Dict[str, Any]] = None):
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    )as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        agent_kwargs: Dict[str, Any] = {
            "name": "Assistant",
            "instructions": ASSISTANT_INSTRUCTIONS,
            "model": OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=external_client,
            )
        }

        if use_tool:
            server_for_agent: MCPServer = mcp_server
            if filter_info and filter_info.get("allowed_names"):
                server_for_agent = MCPServerToolFilterAdapter(
                    mcp_server, set(filter_info["allowed_names"])
                )
            agent_kwargs["mcp_servers"] = [server_for_agent]
        else:
            filter_info = None  # 工具未启用时不显示过滤信息

        agent = Agent(**agent_kwargs)
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        filter_info = resolve_tool_filter(prompt) if use_tool else None
        if use_tool:
            if filter_info:
                st.session_state["tool_filter_status"] = filter_info["label"]
            else:
                st.session_state["tool_filter_status"] = "自动：允许访问全部已接入工具"
        else:
            st.session_state["tool_filter_status"] = "未启用工具调用"

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    response_generator = get_model_response(prompt, model_name, use_tool, filter_info)

                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text

                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error during streaming: {e}")

            # 4. 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

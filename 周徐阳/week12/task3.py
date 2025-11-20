import streamlit as st

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
    tool_selection = st.selectbox(
        "工具选择模式",
        ["自动选择", "仅使用新闻工具", "仅使用普通工具"],
        help="自动根据对话内容或手动限制可用工具范围"
    )


# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)

# 可用工具列表，基于 mcp_server 中的工具名称
NEWS_TOOL_NAMES = [
    "get_today_daily_news",
    "get_douyin_hot_news",
    "get_github_hot_news",
    "get_toutiao_hot_news",
    "get_sports_news",
]

COMMON_TOOL_NAMES = [
    "get_city_weather",
    "get_address_detail",
    "get_tel_info",
    "get_scenic_info",
    "get_flower_info",
    "get_rate_transform",
    "sentiment_classification",
    # 保留一些非新闻类辅助工具
    "get_today_familous_saying",
    "get_today_motivation_saying",
    "get_today_working_saying",
]

NEWS_KEYWORDS = ['新闻', '资讯', '消息', '报道', '头条', '热点', '时事', 'news']


def build_tool_filter(prompt: str, selection: str, tools_enabled: bool):
    """
    根据用户选择与对话内容生成 tool_filter，限制 MCP 可调用的工具。
    """
    if not tools_enabled:
        return None

    prompt_lower = prompt.lower()
    use_news_tools = selection == "仅使用新闻工具" or (
        selection == "自动选择" and any(keyword in prompt_lower for keyword in NEWS_KEYWORDS)
    )

    if use_news_tools:
        return {"allowed_tool_names": NEWS_TOOL_NAMES}
    return {"allowed_tool_names": COMMON_TOOL_NAMES}


async def get_model_response(prompt, model_name, use_tool, tool_selection):
    tool_filter = build_tool_filter(prompt, tool_selection, use_tool)
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20,
            tool_filter=tool_filter,
    )as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): # 用户输入
            st.markdown(prompt)

        with st.chat_message("assistant"): # 大模型输出
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    response_generator = get_model_response(prompt, model_name, use_tool, tool_selection)

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

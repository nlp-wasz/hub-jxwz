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

    # 添加工具选择选项
    tool_selection = st.selectbox(
        "工具选择模式",
        ["自动选择", "仅使用新闻工具", "仅使用普通工具"],
        help="自动选择：根据对话内容自动选择工具；仅使用新闻工具：只调用新闻相关工具；仅使用普通工具：只调用普通功能工具"
    )

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


def should_use_news_tools(prompt, tool_selection):
    """
    判断是否应该使用新闻工具
    """
    if tool_selection == "仅使用新闻工具":
        return True
    elif tool_selection == "仅使用普通工具":
        return False
    else:  # 自动选择
        # 根据关键词判断是否需要新闻工具
        news_keywords = ['新闻', '资讯', '消息', '报道', '头条', '热点', '时事']
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in news_keywords)


def create_agent_instructions(use_news_tools):
    """
    根据工具类型创建相应的指令
    """
    if use_news_tools:
        return "你是一个企业职能助手，专门处理新闻相关的查询。请使用新闻工具来获取最新的新闻资讯。"
    else:
        return "你是一个企业职能助手，请使用普通工具来处理各种企业职能相关的任务。"


async def get_model_response(prompt, model_name, use_tool, tool_selection):
    # 判断是否使用新闻工具
    use_news_tools = should_use_news_tools(prompt, tool_selection)

    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if use_tool:
            # 根据工具类型创建不同的指令
            instructions = create_agent_instructions(use_news_tools)

            agent = Agent(
                name="Assistant",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="你是一个企业职能助手，可以进行普通的对话交流。",
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
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    # 获取当前选择的工具模式
                    current_tool_selection = tool_selection

                    response_generator = get_model_response(prompt, model_name, use_tool, current_tool_selection)


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

            # 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
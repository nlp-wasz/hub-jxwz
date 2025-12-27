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

    if use_tool:
        selected_tool = st.selectbox(
            "请选择工具：",
            ["news", "saying", "tool"],
            help="选择需要启用的工具"
        )
    else:
        selected_tool = None

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


def get_mcp_server_url(tool_type):
    """根据选择的工具类型返回对应的MCP服务器URL"""
    tool_configs = {
        "news": "http://localhost:8901/sse",  # 新闻服务端口
        "saying": "http://localhost:8902/sse",  # 名言服务端口
        "tool": "http://localhost:8900/sse"  # 默认工具服务端口
    }
    return tool_configs.get(tool_type, "http://localhost:8900/sse")


async def get_model_response(prompt, model_name, use_tool, selected_tool=None):
    if selected_tool:
        mcp_url = get_mcp_server_url(selected_tool)
        tool_prompt = f"（使用{selected_tool}服务）"
        yield tool_prompt
        async with MCPServerSse(
                name=f"{selected_tool.upper()} Python Server",
                params={
                    "url": mcp_url,
                },
                client_session_timeout_seconds=20
        ) as mcp_server:
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
    else:
        # 不使用工具的情况
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        agent = Agent(
            name="Assistant",
            instructions="你是一个智能助手，请帮助用户解决问题。",
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

                    if use_tool and selected_tool:
                        st.info(f"正在使用 {selected_tool.upper()} 服务...")
                    response_generator = get_model_response(prompt, model_name, use_tool)


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

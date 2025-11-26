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
session = SQLiteSession("conversation_123")  # openai agent 提供的 基于内存的上下文缓存

# streamlit
# session_state 当前对话的缓存
# session_state.messages 此次对话的历史上下文

# 页面的侧边栏
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

    # 新增：工具选择功能 ###
    if use_tool:
        tool_option = st.selectbox("选择工具", ["查询新闻", "调用工具"], index=0)
        st.session_state['TOOL_OPTION'] = tool_option

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


async def get_model_response(prompt, model_name, use_tool):
    """
    prompt 当前用户输入
    model_name 模型版本
    use_tool 是否调用工具
    """
    # 新增：获取工具选项 ###
    tool_option = st.session_state.get('TOOL_OPTION', '查询新闻')

    # # 定义工具过滤器函数 ###
    # def tool_filter(tool_name: str) -> bool:
    #     if tool_option == "查询新闻":
    #         # 只允许news_开头的工具
    #         return tool_name.startswith("news/")
    #     elif tool_option == "调用工具":
    #         # 只允许tool_开头的工具
    #         return tool_name.startswith("tool/")
    #     else:
    #         # 默认情况下不限制
    #         return True

    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            cache_tools_list=False, # 如果 True 第一次调用后，缓存mcp server 所有工具信息，不再进行list tool
            # tool_filter 对tool筛选（可以写一个函数筛选，也可以通过黑名单/白名单筛选）
            # tool_filter=tool_filter,
            # client_session_timeout_seconds 超时时间
            client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if use_tool:
            # 新增：根据工具选项调整指令 ###
            instructions = f"用户选择了工具: {tool_option}。如果用户提问与该类型不符，告知用户选择适当的工具再提问。否则请使用这个类型的MCP工具来回答问题。"

            agent = Agent(
                name="Assistant",
                instructions="",
                # instructions=instructions,
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

        # session openai-agent 中 缓存的上下文
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

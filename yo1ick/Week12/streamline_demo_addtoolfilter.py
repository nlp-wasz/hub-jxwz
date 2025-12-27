import streamlit as st
import re

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


def tool_filter(tools, prompt):
    """
    根据用户输入和场景筛选可用工具
    tools: MCP服务器返回的工具列表
    prompt: 当前用户输入
    """
    # 新闻相关关键词列表
    news_keywords = ["新闻", "热点", "头条", "资讯", "时事", "报道"]
    # 检查是否为新闻查询场景
    is_news_query = any(re.search(keyword, prompt, re.IGNORECASE) for keyword in news_keywords)
    
    if is_news_query:
        # 仅保留新闻相关工具（名称包含"news"）
        return [tool for tool in tools if "news" in tool["name"].lower()]
    
    # 检查是否为明确工具调用场景（用户勾选使用工具且未触发新闻场景）
    if use_tool:
        # 仅保留工具分类下的工具（排除新闻和名言类工具）
        excluded_keywords = ["news", "saying"]
        return [tool for tool in tools if not any(keyword in tool["name"].lower() for keyword in excluded_keywords)]
    
    # 默认返回所有工具
    return tools


async def get_model_response(prompt, model_name, use_tool):
    """
    prompt 当前用户输入
    model_name 模型版本
    use_tool 是否调用工具
    """
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            cache_tools_list=False,  # 不缓存工具列表，确保过滤规则实时生效
            tool_filter=lambda tools: tool_filter(tools, prompt),  # 应用工具过滤
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

        # session openai-agent 中 缓存的上下文
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):  # 用户输入
            st.markdown(prompt)

        with st.chat_message("assistant"):  # 大模型输出
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

            # 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

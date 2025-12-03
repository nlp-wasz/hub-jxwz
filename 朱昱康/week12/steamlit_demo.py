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
session = SQLiteSession("conversation_123")

# 意图识别函数
def detect_intent(prompt):
    """检测用户意图，返回工具类别"""
    prompt_lower = prompt.lower()
    
    # 新闻相关关键词
    news_keywords = [
        '新闻', '资讯', '消息', '头条', '热点', '今日', '最新', '新闻资讯', 
        '新闻热点', '今日新闻', '最新消息', '头条新闻', '热点新闻',
        'news', 'headline', 'breaking news', 'latest news', 'today news'
    ]
    
    # 工具相关关键词
    tool_keywords = [
        '天气', '查询', '汇率', '地址', '电话', '景点', '花语', '转换',
        'weather', 'exchange rate', 'address', 'phone', 'scenic spot', 'flower language'
    ]
    
    # 名言相关关键词
    saying_keywords = [
        '名言', '名言警句', '励志', '语录', '格言', '鸡汤', '名言名句',
        'quote', 'saying', 'motivation', 'inspiration', 'famous quote'
    ]
    
    # 情感分析相关关键词
    sentiment_keywords = [
        '情感', '情绪', '分析', '感受', '心情', '态度', '看法',
        'sentiment', 'emotion', 'feeling', 'mood', 'attitude', 'analyze'
    ]
    
    # 检查是否包含新闻关键词
    if any(keyword in prompt_lower for keyword in news_keywords):
        return "news"
    
    # 检查是否包含工具关键词
    if any(keyword in prompt_lower for keyword in tool_keywords):
        return "tools"
    
    # 检查是否包含名言关键词
    if any(keyword in prompt_lower for keyword in saying_keywords):
        return "saying"
    
    # 检查是否包含情感分析关键词
    if any(keyword in prompt_lower for keyword in sentiment_keywords):
        return "sentiment"
    
    # 默认返回None，表示不确定意图
    return None

# 获取对应工具类别的MCP服务器URL
def get_mcp_server_url(tool_category):
    """根据工具类别返回对应的MCP服务器URL"""
    if tool_category == "news":
        return "http://localhost:8901/sse"
    elif tool_category == "tools":
        return "http://localhost:8902/sse"
    elif tool_category == "saying":
        return "http://localhost:8903/sse"
    elif tool_category == "sentiment":
        return "http://localhost:8904/sse"
    else:
        return "http://localhost:8900/sse"  # 默认使用主服务器

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
    auto_filter = st.checkbox("智能工具过滤", value=True, help="根据用户意图自动选择合适的工具类别")


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

async def get_model_response(prompt, model_name, use_tool, auto_filter):
    # 如果启用自动过滤，检测用户意图
    if use_tool and auto_filter:
        intent = detect_intent(prompt)
        mcp_url = get_mcp_server_url(intent)
        
        # 在侧边栏显示检测到的意图
        if intent:
            st.sidebar.info(f"检测到意图: {intent}")
        else:
            st.sidebar.info("未检测到明确意图，使用全部工具")
    else:
        mcp_url = "http://localhost:8900/sse"  # 默认使用主服务器
    
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": mcp_url,
            },
            client_session_timeout_seconds=20
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
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    response_generator = get_model_response(prompt, model_name, use_tool, auto_filter)

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

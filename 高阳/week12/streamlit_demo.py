import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
import openai
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import os

# 配置设置
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人 - OpenAI Agent增强版")
session = SQLiteSession("conversation_123")

# 侧边栏配置
with st.sidebar:
    st.title('职能AI+智能问答')

    # API Token配置
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)
    st.session_state['API_TOKEN'] = key

    # 模型选择
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max", "gpt-3.5-turbo", "gpt-4"])

    # 功能开关
    use_tool = st.checkbox("使用MCP工具", value=True)
    use_openai_agent = st.checkbox("启用OpenAI Agent", value=True)

    # OpenAI Agent配置
    if use_openai_agent:
        st.subheader("OpenAI Agent设置")
        agent_temperature = st.slider("Agent创造力", 0.0, 1.0, 0.7)
        agent_memory_length = st.slider("记忆长度", 5, 20, 10)

# 初始化对话历史
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant",
         "content": "你好，我是企业职能助手，具备OpenAI Agent增强能力，可以AI对话、调用工具和进行复杂任务处理。"}
    ]

# 初始化OpenAI Agent记忆
if "openai_memory" not in st.session_state:
    st.session_state.openai_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_length=agent_memory_length if 'agent_memory_length' in locals() else 10
    )

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    """清空聊天历史"""
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，具备OpenAI Agent增强能力。"}
    ]
    st.session_state.openai_memory.clear()
    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


def create_openai_agent_tools():
    """创建OpenAI Agent工具集"""
    tools = [
        Tool(
            name="CurrentTime",
            func=lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description="获取当前日期和时间"
        ),
        Tool(
            name="Calculator",
            func=lambda x: str(eval(x)),
            description="进行数学计算，输入数学表达式"
        ),
        Tool(
            name="TextAnalyzer",
            func=lambda x: f"分析文本: {x}，字符数: {len(x)}",
            description="分析文本内容和统计信息"
        )
    ]
    return tools


async def get_openai_agent_response(prompt, model_name, temperature):
    """OpenAI Agent响应处理"""
    try:
        # 设置OpenAI API
        openai.api_key = key

        # 创建工具
        tools = create_openai_agent_tools()

        # 初始化OpenAI模型
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=key
        )

        # 创建Agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=st.session_state.openai_memory,
            verbose=True,
            handle_parsing_errors=True
        )

        # 执行Agent
        response = agent.run(input=prompt)
        return response

    except Exception as e:
        return f"OpenAI Agent处理错误: {str(e)}"


async def get_model_response(prompt, model_name, use_tool, use_openai_agent):
    """获取模型响应 - 集成OpenAI Agent"""

    # 如果启用OpenAI Agent且使用qwen-max模型
    if use_openai_agent:
        result = get_openai_agent_response(prompt,model_name,0.7)
        async for event in result:
            yield event.data.delta

    # 原有的MCP工具调用逻辑
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
            agent = Agent(
                name="Assistant",
                instructions="你是一个智能的企业职能助手，可以帮助用户解决各种问题。",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="你是一个智能的企业职能助手，可以帮助用户解决各种问题。",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


# 主聊天逻辑
if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("AI正在思考..."):
                try:
                    # 获取响应生成器
                    response_generator = get_model_response(
                        prompt, model_name, use_tool,
                        use_openai_agent if 'use_openai_agent' in st.session_state else False
                    )


                    # 异步流式处理
                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text


                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"请求处理失败: {str(e)}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Streaming error: {e}")

            # 添加助手回复到会话状态
            st.session_state.messages.append({"role": "assistant", "content": full_response})

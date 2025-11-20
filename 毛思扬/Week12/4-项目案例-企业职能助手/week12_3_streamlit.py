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
    # use_tool = st.checkbox("使用工具")

    # 添加工具选择多选框
    tool_categories = st.multiselect(
        "选择工具类别",
        ["news", "tools", "saying", "sentiment"],
        default=["news", "tools", "saying", "sentiment"],
        help="选择要启用的工具类别"
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

async def build_mcp_servers(categories):
    CATEGORY_TO_PORT = {
        "news": 8900,
        "tools": 8903,
        "saying": 8901,
        "sentiment": 8902
    }

    servers = []
    for category in categories:
        port = CATEGORY_TO_PORT.get(category)
        if port:
            try:
                # 更严格的连接检查
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()

                if result == 0:
                    server = MCPServerSse(
                        name=f"{category.capitalize()} MCP Server",
                        params={
                            "url": f"http://localhost:{port}/sse",
                        },
                        client_session_timeout_seconds=20
                    )
                    # 异步调用 connect 方法
                    await server.connect()
                    servers.append(server)
                    print(f"成功连接到 {category} 服务 (端口 {port})")
                else:
                    print(f"跳过 {category} 服务 (端口 {port} 未响应)")
            except Exception as e:
                print(f"跳过 {category} 服务: {e}")
    return servers
async def get_model_response(prompt, model_name, tool_categories):
    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print("tool_categories:", tool_categories)

    # 为整个操作设置超时
    try:
        async with asyncio.timeout(30):  # 30秒总超时
            if tool_categories:
                # 根据选择的工具类别创建相应的MCP Server列表
                mcp_servers = await build_mcp_servers(tool_categories)
                print(mcp_servers)
                instructions = f"""
                            你是一个企业职能助手，可以根据用户需求使用以下工具类别：{', '.join(tool_categories)}。
                            当用户提出相关问题时，请主动调用相应工具获取准确信息。
                            """
                # 创建Agent并传入选定的MCP Servers
                agent = Agent(
                    name="Assistant",
                    instructions=instructions,
                    mcp_servers=mcp_servers,
                    model=OpenAIChatCompletionsModel(
                        model=model_name,
                        openai_client=external_client,
                    )
                )
            else:
                # 不使用工具
                agent = Agent(
                    name="Assistant",
                    instructions="你是一个企业职能助手，请直接回答用户问题。",
                    model=OpenAIChatCompletionsModel(
                        model=model_name,
                        openai_client=external_client,
                    )
                )

            result = Runner.run_streamed(agent, input=prompt, session=session)

            # 改进的事件处理逻辑
            try:
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        yield event.data.delta
                    # 可以添加其他事件类型的处理（如果需要）
            except Exception as e:
                # 添加异常处理，确保生成器能正常结束
                print(f"Stream error: {e}")
                yield f"[Error: {str(e)}]"

    except asyncio.TimeoutError:
        yield "[Error: 请求超时，请检查工具服务是否正常运行]"
    except Exception as e:
        print(f"Run error: {e}")
        yield f"[Error: {str(e)}]"


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
                    # 直接运行异步生成器，避免嵌套事件循环问题
                    async def stream_and_accumulate():
                        accumulated_text = ""
                        try:
                            # 直接调用异步生成器
                            async for chunk in get_model_response(prompt, model_name, tool_categories):
                                # 确保chunk是字符串类型
                                if isinstance(chunk, bytes):
                                    chunk = chunk.decode('utf-8')
                                accumulated_text += chunk
                                message_placeholder.markdown(accumulated_text + "▌")
                        except UnicodeDecodeError as e:
                            # 处理编码错误
                            print(f"Unicode decode error: {e}")
                        return accumulated_text

                    # 使用 asyncio.new_event_loop() 来创建新的事件循环
                    loop = asyncio.new_event_loop()
                    full_response = loop.run_until_complete(stream_and_accumulate())
                    loop.close()
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error during streaming: {e}")

            # 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

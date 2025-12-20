# 自定义 streamlit web 界面
import streamlit as st, os, asyncio
from mcp_server import byMcpTypeGetTools, byMcpTypeGetToolsAuto  # 获取 mcp_tools
from agents import (
    Agent, Runner, set_tracing_disabled, set_default_openai_api,
    AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
)
from agents.mcp import MCPServerSse
from openai.types.responses import ResponseTextDeltaEvent

# 环境配置
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 页面配置
st.set_page_config(page_title="企业智能助手", layout="wide")

# 页面 侧边栏
sidebar = st.sidebar
with sidebar:
    # 输入框 记录key（配置了 key参数，自动将其加载到 st.session_state）
    api_key = st.text_input(label="输入Token:", type="password", key="API_KEY")

    # 等价于 st.session_state.get("API_KEY")
    if api_key and len(api_key) > 1:
        # 获取 key
        st.success("API_KEY 已配置")
    else:
        st.error("API_KEY 尚未配置")

    # 是否使用mcp工具
    is_mcp_tools = st.radio(label="是否使用mcp工具", options=[False, True], index=0)
    if is_mcp_tools:
        # 下拉选项（选择 哪类工具：新闻、情感分析、通用、全部等）
        mcp_tools_type = st.selectbox(label="工具类型", options=["all", "news", "saying", "tool", "emotion"], index=0)

    # 模型选择
    model_name = st.selectbox(label="模型", options=["qwen-max", "qwen-flash"], index=0)

    # 清空聊天记录
    if st.button("清空聊天记录"):
        # 清空 st.session_state.message
        st.session_state["message"] = [
            {"role": "assistant", "content": "欢迎使用企业智能助手！"}
        ]

        # 清空 SqliteSession Agent缓存
        st.session_state["sqlite_session"] = SQLiteSession("session1")

# 判断是否存在聊天信息
if "message" not in st.session_state:
    # 自定义一条信息，用于展示
    st.session_state["message"] = [
        {"role": "assistant", "content": "欢迎使用企业智能助手！"}
    ]

# SqliteSession Agent缓存
if "sqlite_session" not in st.session_state:
    st.session_state["sqlite_session"] = SQLiteSession("session1")

# 聊天内容展示框
for mess in st.session_state["message"]:
    with st.chat_message(mess["role"]):
        st.write(mess["content"])


# 异步执行Agent（生成器，流式输出）
async def get_agent_res(question):
    asyncOpenAI = AsyncOpenAI(
        api_key=api_key,  # sk-04ab3d7290e243dda1badc5a1d5ac858
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=asyncOpenAI
    )

    # 根据选择的模型，过滤出对应的 mcp_tools
    mcp_type_tools = byMcpTypeGetToolsAuto(mcp_tools_type)

    # mcp_type_tools = await byMcpTypeGetTools(mcp_tools_type)

    def get_tools(tool_filter_context, tool_meta) -> bool:
        return tool_meta.name in mcp_type_tools

    # MCP 服务端连接
    mcp_sse = MCPServerSse(
        name="MCP_SSE",
        params={"url": "http://127.0.0.1:8000/sse"},
        client_session_timeout_seconds=20,
        tool_filter=get_tools,  # 根据 工具类型，选择特定mcp工具
    )

    # 根据是否使用 mcp工具，使用不同的 Agent
    async with mcp_sse:
        # 调用 Agent 回答用户问题
        agent = Agent(
            name="企业智能助手",
            model=model,
            instructions="""你是一个专业的企业智能助手，请认真记住用户在对话中主动提供的个人信息（如姓名、年龄、公司等），
            并在后续对话中准确引用。不要编造信息，如果不确定，请反问用户。""",
            mcp_servers=[mcp_sse] if is_mcp_tools else []
        )

        # 执行 Agent
        runner_res = Runner.run_streamed(agent, question, session=st.session_state["sqlite_session"])
        # for i in runner_res.new_items:
        #     print(i)
        async for event in runner_res.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


# 聊天框 chat_text()
if api_key and len(api_key) > 1:
    # 展示 聊天框
    if question := st.chat_input():
        # 将用户提问 存储到 st.session_state.message
        st.session_state["message"].append({"role": "user", "content": question})
        # 立即展示 用户提问信息
        with st.chat_message("user"):
            st.write(question)

        # 调用 异步函数，获取 Agent结果（生成器对象）
        runner_res = get_agent_res(question)

        with st.chat_message("assistant"):
            # 流式展示Agent结果（需要使用 st.empty() 在页面上实时展示）
            stream_res_list = [""]
            stream_flush = st.empty()

            try:
                # 异步函数，消费生成器对象内容
                async def consume_yield():
                    # 遍历生成器对象
                    async for chunk in runner_res:
                        stream_res_list[0] += chunk
                        # 页面刷新
                        stream_flush.markdown(stream_res_list[0] + "▌")

                    return stream_res_list[0]


                # 执行异步函数，消费生成器对象内容
                stream_res = asyncio.run(consume_yield())
                # 生成器结束后，页面刷新
                stream_flush.markdown(stream_res)

                # 将获取到的结果 存储到 st.session_state
                st.session_state["message"].append({"role": "assistant", "content": stream_res})
            except Exception as e:
                error_msg = f"❌ 请求失败: {e}"
                stream_flush.error(error_msg)
                st.session_state["message"].append({"role": "assistant", "content": error_msg})

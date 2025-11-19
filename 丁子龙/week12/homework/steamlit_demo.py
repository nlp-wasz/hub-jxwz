import streamlit as st
import asyncio
from agents.mcp.server import MCPServerSse
import asyncio
import inspect
from agents import (Agent, Runner, AsyncOpenAI, FunctionTool,
                    OpenAIChatCompletionsModel,RunContextWrapper,Tool,AgentHooks,
                    SQLiteSession,StopAtTools,TContext)
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
from typing import Optional,Callable,List
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")

class MyAgentHooks(AgentHooks):
    async def on_tool_start(self, context, agent, tool):
        # ✅ 获取工具名
        tool_name = getattr(tool, 'name', 'unknown')
        print(f"▶️ Agent 即将调用工具: {tool_name}")


    async def on_tool_end(self, context, agent, tool, result):
        tool_name = getattr(tool, 'name', 'unknown')
        print(f"✅ 工具执行完成: {tool_name} → {result[:100]}...")




class FilteredAgent(Agent[TContext]):
    def __init__(
            self,
            *args,
            allowed_mcp_tool_names: Optional[List[str]] = None,  # ✅ 推荐：明确的白名单
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._allowed_mcp_tool_names = set(allowed_mcp_tool_names) if allowed_mcp_tool_names else None

    async def get_all_tools(self, run_context: RunContextWrapper[TContext]) -> List[Tool]:
        # 获取本地已启用的工具（不变）
        local_enabled = await self._get_enabled_local_tools(run_context)

        # 获取原始 MCP 工具
        mcp_tools = await self.get_mcp_tools(run_context)

        # ✅ 按白名单过滤
        if self._allowed_mcp_tool_names is not None:
            mcp_tools = [
                tool for tool in mcp_tools
                if getattr(tool, 'name', None) in self._allowed_mcp_tool_names
            ]
            print("Filtered MCP tools:", [tool.name for tool in mcp_tools])

        return mcp_tools + local_enabled

    async def _get_enabled_local_tools(self, run_context: RunContextWrapper[TContext]) -> List[Tool]:
        # 复用父类中的启用逻辑（如你前面源码所示）
        async def _check_tool_enabled(tool: Tool) -> bool:
            if not isinstance(tool, FunctionTool):
                return True
            attr = tool.is_enabled
            if isinstance(attr, bool):
                return attr
            res = attr(run_context, self)
            if inspect.isawaitable(res):
                return bool(await res)
            return bool(res)

        results = await asyncio.gather(*(_check_tool_enabled(t) for t in self.tools))
        return [t for t, ok in zip(self.tools, results) if ok]

# 🔁 封装异步调用（关键！）
async def _fetch_tools():
    from fastmcp import Client
    # 注意：SSE 端点应为 /mcp/sse（标准 FastMCP 路径）
    # 若你服务挂载在 /sse，请确保与后端一致
    async with Client("http://localhost:8900/sse") as client:
        tools = await client.list_tools()
        return [tool.name for tool in tools]
tool_names = []
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
        tool_names = asyncio.run(_fetch_tools())
        selected_list = st.multiselect(
            "选择要使用的工具，可多选 👇",
            options=tool_names,
            default=None
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

async def get_model_response(prompt, model_name, use_tool):
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
        if use_tool:
            # tools = asyncio.run(_fetch_tools())
            # abandon_tools = [tool for tool in tool_names if tool not in selected_list]

            agent = FilteredAgent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                allowed_mcp_tool_names = selected_list,
                # stop_at_tool_names = abandon_tools,
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                hooks = MyAgentHooks(),  # ← 注册钩子
            )
        else:
            agent = FilteredAgent(
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

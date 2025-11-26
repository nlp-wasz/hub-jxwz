import streamlit as st
import re
import json

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
import openai
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)
#
st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")


# 意图识别函数
def detect_intent(prompt):
    """根据用户意图，选择工具"""

    client = openai.OpenAI(
        api_key="sk-fad1550b59d547ee83006bde2452e7bc",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本进行意图识别；
        - 待选的意图类别：查询新闻 / 名人名言 / 情感分类 / 调用工具
        最终输出格式填充下面的json，intent 是 意图标签。

        ```json
        {
            "intent": 
        }
        ```
        """},
            {"role": "user", "content": f"{prompt}"},

        ],
    )
    content = completion.choices[0].message.content
    # print(content)
    # 提取 JSON 部分
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            intent = data.get("intent")
            # print(f"提取的意图: {intent}")
            # 返回意图
            return intent
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
    # 表示不确定意图
    return None


def generate_unified_instruction(intent: str, allowed_tools: list) -> str:
    """生成统一的智能指令"""

    tool_descriptions = {
        "get_today_daily_news": "今日新闻简报",
        "get_douyin_hot_news": "抖音热点",
        "get_github_hot_news": "GitHub热门项目",
        "get_toutiao_hot_news": "头条热点",
        "sentiment_classification": "情感分析",
        "get_city_weather": "城市天气查询",
        "get_today_famous_saying": "名人名言",
        "get_today_motivation_saying": "励志语录"
    }

    # 生成工具描述
    tool_desc_list = [tool_descriptions.get(tool, tool) for tool in allowed_tools]
    tools_str = "、".join(tool_desc_list)

    instruction = f"""你是一个智能助手，检测到用户意图是：{intent}。

当前专注领域：{intent}相关服务
可用工具：{tools_str}

请严格按照以下规则工作：
1. 只使用上述可用工具来帮助用户
2. 根据工具返回的结果，提供准确、有用的回复
3. 如果用户请求超出当前工具范围，请礼貌说明
4. 回复要简洁明了，直接回答问题

示例：
用户：今天有什么新闻？
你：调用新闻工具 → 整理新闻内容 → 回复今日热点

用户：分析这句话的情感
你：调用情感分析工具 → 返回情感类型 → 简要说明"""

    return instruction


#
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
    tool_filter = st.checkbox("智能工具过滤", value=True, help="根据用户意图自动选择合适的工具类别")

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


async def get_model_response(prompt, model_name, use_tool, tool_filter):
    allowed_tools = ["get_today_daily_news", "get_douyin_hot_news", "get_github_hot_news", "get_toutiao_hot_news",
                     "sentiment_classification", "get_city_weather", "get_today_famous_saying",
                     "get_today_motivation_saying"]
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=30
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30.0
        )
    # 如果启用自动过滤，检测用户意图
    if tool_filter:
        intent = detect_intent(prompt)
        instruction = generate_unified_instruction(intent, allowed_tools)

        # 在侧边栏显示检测到的意图
        if intent:
            st.sidebar.info(f"检测到意图: {intent}")
        else:
            st.sidebar.info("未检测到明确意图，使用全部工具")
        agent = Agent(
            name="Assistant",
            instructions=instruction,
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=external_client,
            )
        )
    elif use_tool:
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
                    response_generator = get_model_response(prompt, model_name, use_tool, tool_filter)


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

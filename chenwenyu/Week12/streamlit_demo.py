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

def detect_tool_and_params(prompt):
    """检测用户输入应该调用哪个工具以及参数"""
    prompt_lower = prompt.lower()
    
    # 工具映射配置
    tool_mappings = [
        {
            "keywords": ["新闻", "头条", "热点", "资讯", "今日要闻", "最新消息"],
            "tools": [
                {"name": "get_today_daily_news", "params": {}},
                {"name": "get_douyin_hot_news", "params": {}},
                {"name": "get_toutiao_hot_news", "params": {}},
                {"name": "get_sports_news", "params": {}},
                {"name": "get_github_hot_news", "params": {}}
            ]
        },
        {
            "keywords": ["天气", "气温", "温度", "下雨", "晴天", "气象"],
            "tools": [
                {
                    "name": "get_city_weather", 
                    "params": {"city_name": "beijing"},  # 默认值
                    "param_extractor": lambda p: extract_city_name(p)  # 城市名提取函数
                }
            ]
        },
        {
            "keywords": ["汇率", "兑换", "美元", "人民币", "货币", "换算"],
            "tools": [
                {
                    "name": "get_rate_transform",
                    "params": {"source_coin": "USD", "aim_coin": "CNY", "money": 100},
                    "param_extractor": lambda p: extract_currency_params(p)
                }
            ]
        },
        {
            "keywords": ["分类", "归类", "文本分类", "类别"],
            "tools": [
                {
                    "name": "text_classification",
                    "params": {"text": ""},
                    "param_extractor": lambda p: extract_classification_text(p)
                }
            ]
        }
    ]
    
    # 遍历所有工具映射，找到匹配的
    for mapping in tool_mappings:
        if any(keyword in prompt_lower for keyword in mapping["keywords"]):
            # 返回第一个匹配的工具（可以根据需要调整逻辑）
            tool_config = mapping["tools"][0]
            
            # 如果有参数提取函数，使用它来更新参数
            if "param_extractor" in tool_config:
                extracted_params = tool_config["param_extractor"](prompt)
                if extracted_params:
                    tool_config["params"].update(extracted_params)
            
            return tool_config
    
    return None

def extract_city_name(prompt):
    """从文本中提取城市名称"""
    # 简单的城市名映射
    city_mapping = {
        "北京": "beijing", "上海": "shanghai", "广州": "guangzhou", 
        "深圳": "shenzhen", "成都": "chengdu", "杭州": "hangzhou",
        "重庆": "chongqing", "武汉": "wuhan", "西安": "xian"
    }
    
    for chinese_name, pinyin_name in city_mapping.items():
        if chinese_name in prompt:
            return {"city_name": pinyin_name}
    
    # 如果没有匹配的中文城市名，尝试提取可能的拼音
    words = prompt.split()
    for word in words:
        if word.isalpha() and len(word) > 2:  # 简单的拼音检测
            return {"city_name": word.lower()}
    
    return {"city_name": "beijing"}  # 默认返回北京

def extract_currency_params(prompt):
    """提取货币转换参数"""
    import re
    
    # 提取金额
    money_match = re.search(r'(\d+(?:\.\d+)?)\s*(美元|美金|usd)', prompt, re.IGNORECASE)
    if money_match:
        money = float(money_match.group(1))
        return {"money": money, "source_coin": "USD", "aim_coin": "CNY"}
    
    return {"money": 100, "source_coin": "USD", "aim_coin": "CNY"}  # 默认值

def extract_classification_text(prompt):
    """提取需要分类的文本"""
    # 尝试从引号中提取文本
    import re
    text_match = re.search(r'["“”]([^"“”]+)["“”]', prompt)
    if text_match:
        return {"text": text_match.group(1)}
    
    # 如果没有引号，尝试从"文本："后面提取
    if "文本：" in prompt:
        text = prompt.split("文本：")[1].strip()
        return {"text": text}
    
    # 默认返回原提示词
    return {"text": prompt}

async def get_model_response(prompt, model_name, use_tool):
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    )as mcp_server:
        
        if use_tool:
            # 检测应该调用哪个工具
            tool_config = detect_tool_and_params(prompt)
            
            if tool_config:
                try:
                    # 直接调用工具
                    tool_result = await mcp_server.call_tool(
                        tool_config["name"], 
                        tool_config["params"]
                    )
                    
                    # 格式化工具返回结果
                    if hasattr(tool_result, 'content') and tool_result.content:
                        # 从工具结果中提取文本内容
                        content_parts = []
                        for content in tool_result.content:
                            if hasattr(content, 'text'):
                                content_parts.append(content.text)
                        
                        tool_response = "\n".join(content_parts) if content_parts else str(tool_result)
                    else:
                        tool_response = str(tool_result)
                    
                    # 返回工具调用结果
                    yield f"🔧 工具调用: {tool_config['name']}\n\n"
                    yield f"📊 结果: {tool_response}"
                    return
                    
                except Exception as e:
                    yield f"❌ 工具调用失败: {str(e)}\n\n"
                    # 工具调用失败后回退到普通AI对话
            
            else:
                yield "⚠️ 未找到匹配的工具，使用AI对话模式\n\n"

            external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
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

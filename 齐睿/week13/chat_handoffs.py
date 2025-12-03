"""
Agent Handoffs 聊天页面
支持 ChatAgent 和 StockAgent 之间的智能转接
"""

import streamlit as st
import requests
import traceback
import json
import asyncio
from datetime import datetime
from typing import List
import pandas as pd
import plotly.graph_objects as go
from fastmcp import Client
from fastmcp.tools import Tool

# -------------------- 配置 --------------------
BASE_URL = "http://127.0.0.1:8000"
MCP_SERVER_URL = "http://127.0.0.1:8900/sse"
# ----------------------------------------------

st.set_page_config(
    page_title="Agent Handoffs 聊天",
    page_icon="🤖",
    layout="wide"
)

# -------------------- 工具加载 --------------------
@st.cache_data(show_spinner="正在连接 FastMCP 服务器并获取工具列表...", ttl=60)
def load_mcp_tools(url: str) -> tuple[bool, List[Tool]]:
    """
    同步函数中运行异步客户端逻辑，获取所有可用工具。
    """
    async def get_data():
        client = Client(url)
        try:
            async with client:
                ping_result = await client.ping()
                tools_list = await client.list_tools()
                return ping_result, tools_list
        except Exception as e:
            st.error(f"连接 FastMCP 服务器失败或发生错误: {e}")
            traceback.print_exc()
            return False, []
    
    return asyncio.run(get_data())

# -------------------- K线数据获取和绘图 --------------------
def fetch_k_line_data(endpoint: str, code: str, line_type: int, start_date: str, end_date: str):
    """获取K线数据"""
    url = f"{BASE_URL}/stock/{endpoint}"
    params = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": line_type
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            if not df.empty and "date" in df.columns:
                df.rename(columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }, inplace=True)
                return df
        return None
    except Exception as e:
        st.error(f"获取K线数据失败: {str(e)}")
        traceback.print_exc()
        return None

def plot_candlestick(df: pd.DataFrame, stock_code: str, line_type: int):
    """绘制K线图"""
    type_names = {0: "不复权", 1: "前复权", 2: "后复权"}
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线'
    )])
    
    fig.update_layout(
        title=f"股票 K 线图 - {stock_code} ({type_names.get(line_type, '未知')})",
        xaxis_rangeslider_visible=False,
        xaxis=dict(title='日期'),
        yaxis=dict(title='价格'),
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 成交量图
    fig_volume = go.Figure(data=[go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='成交量'
    )])
    
    fig_volume.update_layout(
        title="成交量",
        xaxis=dict(title='日期'),
        yaxis=dict(title='成交量'),
        height=200
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

# -------------------- 聊天请求 --------------------
def request_chat_handoffs(content: str, user_name: str, session_id: str, task: str, selected_tools: list):
    """发送聊天请求到 handoffs 端点"""
    url = f"{BASE_URL}/v1/chat/handoffs"
    
    headers = {
        "accept": "text/event-stream",
        "Content-Type": "application/json"
    }
    
    data = {
        "content": content,
        "user_name": user_name,
        "session_id": session_id,
        "task": task,
        "stream": True,
        "tools": selected_tools
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=300)
        response.raise_for_status()
        for chunk in response.iter_content(decode_unicode=True, chunk_size=1024):
            if chunk:
                yield chunk
    except requests.exceptions.ChunkedEncodingError as e:
        print(f"ChunkedEncodingError: {e}")
        traceback.print_exc()
        yield f"\n\n[错误] 响应流中断: {str(e)}"
    except requests.exceptions.Timeout as e:
        print(f"Timeout: {e}")
        traceback.print_exc()
        yield f"\n\n[错误] 请求超时: {str(e)}"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        yield f"\n\n[错误] 请求失败: {str(e)}"

# -------------------- 主界面 --------------------
st.title("🤖 Agent Handoffs 智能对话")
st.markdown("""
这个页面展示了 **Agent Handoffs** 功能：
- 🗣️ **ChatAgent**：默认处理所有对话
- 📈 **StockAgent**：专门处理股票分析和金融问题
- 🔄 **智能转接**：ChatAgent 会自动识别股票相关问题并转接给 StockAgent

**工作流程**：
1. 所有对话默认由 ChatAgent 处理
2. 当你询问股票、财务分析等问题时，ChatAgent 会自动转接给 StockAgent
3. StockAgent 完成回答后，如果你继续闲聊，会转接回 ChatAgent
""")

# -------------------- 侧边栏配置 --------------------
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 用户信息
    if 'user_name' not in st.session_state:
        st.session_state['user_name'] = 'test_user'
    
    user_name = st.text_input("用户名", value=st.session_state['user_name'])
    st.session_state['user_name'] = user_name
    
    # Session ID
    if 'handoffs_session_id' not in st.session_state:
        st.session_state['handoffs_session_id'] = f"handoffs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_id = st.text_input("Session ID", value=st.session_state['handoffs_session_id'])
    st.session_state['handoffs_session_id'] = session_id
    
    # 默认任务类型为闲聊，由 ChatAgent 自动决定是否转接
    task = "闲聊"
    
    st.info("💡 提示：所有对话默认由 ChatAgent 处理，遇到股票问题会自动转接给 StockAgent")
    
    # 工具选择
    st.subheader("🛠️ 可用工具")
    ping_status, all_tools = load_mcp_tools(MCP_SERVER_URL)
    
    if not ping_status or not all_tools:
        st.error("未能加载工具。请检查 MCP 服务器是否已在 8900 端口运行。")
        selected_tools = []
    else:
        tool_names = [tool.name for tool in all_tools]
        selected_tools = st.multiselect(
            "选择工具（可选）",
            tool_names,
            help="选择需要使用的工具，留空则不使用工具"
        )
    
    # 清空对话
    if st.button("🗑️ 清空对话历史"):
        st.session_state['handoffs_messages'] = []
        st.rerun()

# -------------------- 对话历史 --------------------
if 'handoffs_messages' not in st.session_state:
    st.session_state['handoffs_messages'] = []

# 显示对话历史
for msg in st.session_state['handoffs_messages']:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- 用户输入 --------------------
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state['handoffs_messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 获取 AI 响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        handoff_info = []
        tool_calls = []
        
        try:
            # 同步方式处理流式响应
            for chunk in request_chat_handoffs(
                content=prompt,
                user_name=user_name,
                session_id=session_id,
                task=task,
                selected_tools=selected_tools
            ):
                full_response += chunk
                
                # 检测 Agent 转接信息
                if "🔄 Agent 转接" in chunk or "handoff" in chunk.lower():
                    handoff_info.append(chunk)
                
                # 检测工具调用
                if "```json" in chunk:
                    tool_calls.append(chunk)
                
                # 实时显示
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 显示转接信息
            if handoff_info:
                with st.expander("🔄 Agent 转接记录", expanded=True):
                    for info in handoff_info:
                        st.info(info)
            
            # 处理工具调用（K线图）
            if tool_calls:
                for tool_call in tool_calls:
                    try:
                        # 提取 JSON
                        if "```json" in tool_call:
                            json_start = tool_call.find("{")
                            json_end = tool_call.rfind("}") + 1
                            if json_start != -1 and json_end > json_start:
                                json_str = tool_call[json_start:json_end]
                                
                                # 提取工具名
                                tool_name_start = tool_call.find("```json") + 7
                                tool_name_end = tool_call.find(":", tool_name_start)
                                tool_name = tool_call[tool_name_start:tool_name_end].strip()
                                
                                # 解析参数
                                argv = json.loads(json_str)
                                
                                # 处理 K线工具
                                if tool_name in ["get_month_line", "get_week_line", "get_day_line"]:
                                    endpoint_map = {
                                        "get_month_line": "get_month_line",
                                        "get_week_line": "get_week_line",
                                        "get_day_line": "get_day_line"
                                    }
                                    endpoint = endpoint_map.get(tool_name)
                                    
                                    stock_code = argv.get("code")
                                    start_date_str = argv.get("startDate")
                                    end_date_str = argv.get("endDate")
                                    line_type = argv.get("type", 0)
                                    
                                    if not stock_code:
                                        st.error("❌ 错误：缺少股票代码 (code)")
                                    elif not start_date_str or not end_date_str:
                                        st.warning(f"⚠️ 警告：缺少日期参数。startDate={start_date_str}, endDate={end_date_str}")
                                    else:
                                        with st.spinner(f"正在加载 {stock_code} 数据..."):
                                            df_k_line = fetch_k_line_data(
                                                endpoint=endpoint,
                                                code=stock_code,
                                                line_type=line_type,
                                                start_date=start_date_str,
                                                end_date=end_date_str
                                            )
                                            
                                            if df_k_line is not None and not df_k_line.empty:
                                                st.success(f"✅ 成功加载 {len(df_k_line)} 条数据")
                                                plot_candlestick(df_k_line, stock_code, line_type)
                                            else:
                                                st.info("没有数据可以绘制 K 线图")
                    except Exception as e:
                        st.error(f"处理工具调用时出错: {str(e)}")
                        traceback.print_exc()
            
        except Exception as e:
            st.error(f"请求失败: {str(e)}")
            traceback.print_exc()
            full_response = f"抱歉，请求失败: {str(e)}"
        
        # 保存助手消息
        st.session_state['handoffs_messages'].append({"role": "assistant", "content": full_response})

# -------------------- 页面底部信息 --------------------
st.markdown("---")
st.markdown("""
### 💡 使用提示

**测试智能转接**：
1. 问一个闲聊问题：
   - "你好" → ChatAgent 直接响应
   - "今天天气怎么样" → ChatAgent 直接响应

2. 问一个股票问题：
   - "请帮我查看北京银行的代码" → ChatAgent 自动转接到 StockAgent
   - "分析一下苹果公司的股票" → ChatAgent 自动转接到 StockAgent
   - "请查找sh601169的月K线" → ChatAgent 自动转接到 StockAgent

3. 继续闲聊：
   - "谢谢，今天天气怎么样" → StockAgent 自动转接回 ChatAgent

**观察转接过程**：
- 🔄 转接信息会在对话中显示
- 📊 工具调用会以 JSON 格式显示
- 📈 K线图会自动绘制

**关键特性**：
- ✅ 无需手动选择任务类型
- ✅ Agent 自动识别问题类型
- ✅ 智能转接，无缝切换
""")

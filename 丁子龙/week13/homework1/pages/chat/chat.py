import re
import time
import plotly.graph_objects as go

import requests
import streamlit as st
import asyncio
import traceback
import json
from fastmcp import Client
from fastmcp.tools import Tool
from typing import List, Any
import pandas as pd

# FastMCP 服务器地址
MCP_SERVER_URL = "http://127.0.0.1:8900/sse"


@st.cache_data(show_spinner="正在连接 FastMCP 服务器并获取工具列表...", ttl=60)
def load_mcp_tools(url: str) -> tuple[bool, List[Tool]]:
    """
    同步函数中运行异步客户端逻辑，获取所有可用工具。
    """

    async def get_data():
        client = Client(url)
        try:
            # 使用 async with 确保客户端连接正确管理
            async with client:
                ping_result = await client.ping()
                tools_list = await client.list_tools()
                return ping_result, tools_list
        except Exception as e:
            st.error(f"连接 FastMCP 服务器失败或发生错误: {e}")
            traceback.print_exc()
            return False, []

    return asyncio.run(get_data())


# streamlit
# session_state 当前对话的缓存
# session_state.messages 此次对话的历史上下文

if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")
else:
    st.info("请先登录再使用模型～")

# 初始化的对话
if "messages" not in st.session_state.keys() or "session_id" in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
    ]
    if "session_id" in st.session_state.keys() and st.session_state.session_id:
        data = requests.post("http://127.0.0.1:8000/v1/chat/get?session_id=" + st.session_state['session_id']).json()
        for message in data["data"]:
            if message["role"] == "system":
                continue
            st.session_state.messages.append({"role": message["role"], "content": message["content"]})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
    ]
    st.session_state.session_id = None


with st.sidebar:
    if st.session_state.get('logged', False):
        ping_status, all_tools = load_mcp_tools(MCP_SERVER_URL)

        if not ping_status or not all_tools:
            st.error("未能加载工具。请检查服务器是否已在 8900 端口运行，并查看上方错误详情。")
            selected_tool_names = []
        else:
            # 将工具列表转换为 {name: Tool} 字典，方便查找
            tool_map = {tool.name: tool for tool in all_tools}
            tool_names = list(tool_map.keys())

            selected_tool_names = st.multiselect(
                "选择MCP工具:",
                options=tool_names,
            )

    st.button('清空当前聊天', on_click=clear_chat_history, use_container_width=True)


async def request_chat(content: str, user_name: str, session_id: str) -> str:
    url = "http://127.0.0.1:8000/v1/chat/"

    headers = {
        "accept": "text/event-stream",  # 修改为接受事件流
        "Content-Type": "application/json"
    }

    data = {
        "content": content.text,
        "user_name": user_name,
        "session_id": session_id,
        "stream": True,
        "tools": selected_tool_names
    }

    if not session_id:
        del data["session_id"]

    response = requests.post(url, headers=headers, json=data, stream=True)
    for content in response.iter_content(decode_unicode=True):
        if content:
            yield content

def request_session_id() -> str:
    url = "http://127.0.0.1:8000/v1/chat/init"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    return response.json()["data"]["session_id"]


def fetch_k_line_data(
        endpoint: str,
        code: str,
        line_type: str,
        start_date: str,
        end_date: str,
        data_type: int = 0  # 假设 type=0 是默认的数据类型
):
    """
    通过调用后端 API 获取 K 线数据。
    """

    BASE_URL = "http://127.0.0.1:8000/stock/"
    url = f"{BASE_URL}{endpoint}"

    # 注意：您的 curl 示例中，日期参数被双引号包裹，但在 Python requests 中，
    # 传递日期字符串通常不需要额外的引号，后端应自行解析。
    params = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": data_type,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            # 假设返回的数据结构是列表的列表：
            # [ ["日期", "昨收", "今开", "最高", "最低", "成交量"], ... ]

            # 转换为 DataFrame
            df = pd.DataFrame(data["data"])
            df = df.iloc[:, :6]
            df.columns=[
                "Date", "Close_Prev", "Open", "High", "Low", "Volume"
            ]

            # 转换为正确的数据类型
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ["Open", "High", "Low", "Close_Prev", "Volume"]:
                # 将数据类型转换为浮点数，并处理可能存在的错误值
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.rename(columns={'Close_Prev': 'Close'}, inplace=True)

            return df
        else:
            st.warning(f"API 返回成功，但未找到 {code} 的 K 线数据。")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"连接错误：无法连接到后端服务 ({BASE_URL})。请确保后端服务正在运行。")
        return None
    except Exception as e:
        st.error(f"获取 K 线数据时发生错误：{e}")
        traceback.print_exc()
        return None


def plot_candlestick(df: pd.DataFrame, code: str, line_type: str):
    """
    使用 Plotly 绘制交互式 K 线图。
    """

    # 确保数据按日期排序
    df = df.sort_values(by='Date')

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线'
    )])

    # 添加成交量 (Volume) 作为子图
    fig_volume = go.Figure(data=[go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='成交量'
    )])

    # 合并图表 (使用 make_subplots 可能会更好，但这里简化为两个独立的图)
    # 调整布局
    fig.update_layout(
        title=f"股票 K 线图 - {code} ({line_type})",
        xaxis_rangeslider_visible=False,  # 隐藏底部的时间轴滑动条
        xaxis=dict(title='日期'),
        yaxis=dict(title='价格'),
        hovermode="x unified",
        height=600  # 增加高度
    )

    # 绘制成交量图（如果需要合并子图，需要使用 plotly.subplots.make_subplots）
    # 在 Streamlit 中，通常将它们分开显示更简单
    st.plotly_chart(fig, use_container_width=True)

    fig_volume.update_layout(
        title="成交量 Volume",
        xaxis=dict(title='日期', showticklabels=True),
        yaxis=dict(title='成交量'),
        height=200
    )
    st.plotly_chart(fig_volume, use_container_width=True)




if prompt := st.chat_input(accept_file="multiple", file_type=["txt", "pdf", "jpg", "png", "jpeg", "doc", "docx"]):
    if "session_id" not in st.session_state.keys() or not st.session_state.session_id:
        st.session_state.session_id = request_session_id()

    if st.session_state.get('logged', False):
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        with st.chat_message("user"):  # 用户输入
            st.markdown(prompt.text)

        with st.chat_message("assistant"):  # 大模型输出
            message_placeholder = st.empty()
            placeholder = st.empty()

            with st.spinner("请求中..."):
                async def stream_output():
                    accumulated_text = ""
                    response_generator = request_chat(prompt, st.session_state['user_name'], st.session_state['session_id'])
                    async for data in response_generator:
                        accumulated_text += data
                        placeholder.markdown(accumulated_text + "▌") # 后端不断sse输出内容，前端通过markdown渲染

                    return accumulated_text

                final_text = asyncio.run(stream_output())
                placeholder.markdown(final_text) # 最后都输出完成了，重新渲染一次

            st.session_state.messages.append({"role": "assistant", "content": final_text})

            try:
                # 如果tool是如下的可视化的工具，则需要调用得到原始数据再进行绘图
                if "get_day_line" in final_text or "get_week_line" in final_text or "get_month_line" in final_text:

                    # 解析工具json
                    function_json = re.search(r"```json\s*([\s\S]*?)\s*```", final_text, re.I).group(1).strip()
                    function_json = function_json.strip()
                    endpoint = function_json[:function_json.index(":")] # 工具名字
                    argv = json.loads(function_json[function_json.index(":")+1:]) # 工具传入参数
                    stock_code = argv["code"]
                    start_date_str = argv["startDate"]
                    end_date_str = argv["endDate"]
                    line_type = argv["type"]
                    with st.spinner(f"正在加载 {stock_code} 数据 ({start_date_str} 至 {end_date_str})..."):
                        df_k_line = fetch_k_line_data(
                            endpoint=endpoint,
                            code=stock_code,
                            line_type=line_type,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )

                        if df_k_line is not None and not df_k_line.empty:
                            st.success(f"成功加载 {len(df_k_line)} 条数据。")
                            plot_candlestick(df_k_line, stock_code, line_type)
                        else:
                            st.info("没有数据可以绘制 K 线图。请检查代码或日期范围。")
            except:
                traceback.print_exc()

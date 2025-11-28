# 聊天窗口（流式输出）
import asyncio, re, plotly.graph_objects as po
import json

import httpx
import pandas as pd
import requests
import streamlit as st
from httpx import AsyncClient
from fastmcp import Client


# 获取 所有MCP工具
async def get_mcp_tools_list():
    try:
        async with Client("http://127.0.0.1:8002/sse") as client:
            tools = await client.list_tools()
            tools_name = [tool.name for tool in tools]

        return tools_name
    except Exception as e:
        return []


# 侧边栏
with st.sidebar:
    # 清空历史记录
    if st.button("清空历史"):
        st.session_state["message"] = []
        # 清除 session_id
        st.session_state["session_id"] = None
        # 清除 is_load_session_message：不加载聊天历史信息
        st.session_state["is_load_session_message"] = None

    # 工具调用（展示 MCP 工具）
    tools_name = asyncio.run(get_mcp_tools_list())
    st.session_state["tools_name"] = tools_name

    select_tools = st.multiselect(label="工具调用", options=tools_name)

st.info(f"select_tools：{select_tools}")
st.info(f"session_id: {st.session_state.get('session_id', '不存在')}")

# 聊天历史展示
if "message" not in st.session_state or not st.session_state["message"]:
    st.session_state["message"] = [
        {"role": "assistant", "content": "欢迎使用ChatBI智能助手！"}
    ]

# 根据 session_id 查询历史记录
if "is_load_session_message" in st.session_state and st.session_state["is_load_session_message"]:
    url = "http://127.0.0.1:8000/v1/chat/get_message_list"
    data = {
        "session_id": st.session_state["session_id"]
    }

    try:
        res = requests.get(url, params=data)

        # 校验 状态码
        res.raise_for_status()

        if res.json()["res_code"] == 200:
            session_message = res.json()["res_result"]

            for mes in session_message:
                # 添加到 st.session_state["message"]中
                st.session_state["message"].append({"role": mes["message_role"], "content": mes["message_content"]})

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到 {url} 服务端！")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

st.session_state["is_load_session_message"] = None

# 循环展示历史记录
for i in st.session_state["message"]:
    with st.chat_message(i["role"]):
        st.write(i["content"])


# 发送聊天请求（后端LLM生成回答）
async def chat_request(prompt, login_user_name, session_id, select_tools):
    # 发送 异步 Http请求
    url = "http://127.0.0.1:8000/v1/chat/"
    header = {}
    data = {
        "prompt": prompt,
        "user_name": login_user_name,
        "session_id": session_id,
        "select_tools": select_tools
    }

    # timeout=httpx.Timeout(60)：后端调用工具时，异步http请求可能会超时
    async with AsyncClient(timeout=httpx.Timeout(60)) as client:
        async with client.stream("POST", url, headers=header, json=data) as response:
            async for chunk in response.aiter_text():
                if chunk:
                    yield chunk


# 生成 session_id（调用 API）
def generator_session_id():
    url = "http://127.0.0.1:8000/v1/chat/generator_session_id/"

    try:
        res = requests.get(url)
        # 校验 状态码
        res.raise_for_status()

        if res.json()["res_code"] == 200:
            return res.json()["res_result"]["session_id"]

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到 {url} 服务端！")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

    return None


# 调用API，获取K线图数据
def get_k_data(tool_name, tool_args):
    # 解析参数信息
    tool_args_code = tool_args["code"]
    tool_args_startDate = tool_args["startDate"]
    tool_args_endDate = tool_args["endDate"]
    tool_args_type = tool_args["type"]

    url = f"http://127.0.0.1:8000/stock/{tool_name}/"
    data = {
        "code": tool_args_code,
        "startDate": tool_args_startDate,
        "endDate": tool_args_endDate,
        "type": tool_args_type
    }

    try:
        res = requests.get(url, params=data)
        # 校验 状态码
        res.raise_for_status()

        if res.json()["code"] == 200:
            k_data_pd = pd.DataFrame(res.json()["data"])
            # 只要前6列
            k_data_pd = k_data_pd.iloc[:, :6]
            # 替换列明
            k_data_pd.columns = [
                "Date", "Close", "Open", "High", "Low", "Volume"
            ]

            # 日期列信息转换（转换为日期类型）
            k_data_pd["Date"] = pd.to_datetime(k_data_pd["Date"], "coerce")
            for col in ["Close", "Open", "High", "Low", "Volume"]:
                k_data_pd[col] = pd.to_numeric(k_data_pd[col], "coerce")

            return k_data_pd
        else:
            st.error(f"API 请求失败：无法绘制K线图")

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到 {url} 服务端！")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")


# 绘制 K线图
def draw_k_line(k_data_pd):
    # 使用 plotly.graph_objects as po  绘制
    k_fig = po.Figure(data=[
        po.Candlestick(
            x=k_data_pd['Date'],
            open=k_data_pd['Open'],
            high=k_data_pd['High'],
            low=k_data_pd['Low'],
            close=k_data_pd['Close'],
            name='股票K线',
            increasing_line_color='green',  # 可简写（旧版兼容）
            decreasing_line_color='red'
        )]
    )
    k_fig.update_layout(
        title="股票K线图",
        title_x=0,
        xaxis_title="日期",
        yaxis_title="价格 (元)",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False
    )
    st.plotly_chart(k_fig, use_container_width=True)

    # 交易量柱状图
    b_fig = po.Figure(data=[
        po.Bar(
            x=k_data_pd['Date'],
            y=k_data_pd['Volume'],
            name='交易量',
            marker_color='steelblue',
        )]
    )
    b_fig.update_layout(
        title="交易量柱状图",
        title_x=0,
        xaxis_title="日期",
        yaxis_title="交易量",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False
    )
    st.plotly_chart(b_fig, use_container_width=True)


# 获取 session_id
if "session_id" not in st.session_state or st.session_state["session_id"] is None:
    # 生成 session_id（调用 API）
    session_id = generator_session_id()
    if session_id is None:
        st.error("session_id生成失败，请重新生成")
    else:
        # 缓存 session_id
        st.session_state["session_id"] = session_id

# 聊天框
if prompt := st.chat_input(placeholder="聊天", accept_file=True, file_type=["jpg", "py"]):
    # 记录用户的提问信息
    st.session_state["message"].append({"role": "user", "content": prompt.text})
    # 手动展示
    with st.chat_message("user"):
        st.write(prompt.text)

    # 大模型回答（调用 chat API，流式输出）
    with st.chat_message("assistant"):
        # 实时刷新
        flush = st.empty()
        llm_res = ""

        with st.spinner("正在请求LLM回答..."):
            # 发送http请求，调用chat API，流式输出（需要使用异步函数接收）
            async def consume_stream():
                global llm_res

                # 获取 API 调用结果（异步生成器 yield返回结果）

                llm_generator = chat_request(prompt.text, st.session_state["login_user_name"],
                                             st.session_state["session_id"], select_tools)

                async for chunk in llm_generator:
                    # 在 页面上实时刷新
                    llm_res += chunk
                    flush.write(f"{llm_res} ▌")


            # 异步执行  consume_stream
            asyncio.run(consume_stream())
            # 手动刷新，展示完整信息
            flush.write(llm_res)

            # st.session_state["message"] 记录信息
            st.session_state["message"].append({"role": "assistant", "content": llm_res})

            with st.spinner("正在绘制K线图"):
                # 判断 调用的工具是否是 查询K线数据的MCP工具（是的话需要绘制K线图）
                # 解析 llm_res（具有 工具返回结果）
                re_llm_res = re.search(r'(\w+):({.*?})', llm_res)

                # 解析成功
                if re_llm_res:
                    # 获取方法名
                    tool_name = re_llm_res.group(1)

                    # 如果方法名如下，则获取参数信息 以及 绘制K线图
                    if tool_name in ["get_stock_month_kline", "get_stock_week_kline", "get_stock_day_kline"]:
                        try:
                            tool_args = re_llm_res.group(2)
                            tool_args = json.loads(tool_args)

                            if isinstance(tool_args, dict):
                                # 发送http，调用API，获取K线图数据
                                k_data_pd = get_k_data(tool_name, tool_args)

                                # 绘制 K线图
                                draw_k_line(k_data_pd)
                            else:
                                st.error(f"参数解析有误TYPE：{re_llm_res.group(2)}")
                        except Exception as e:
                            st.error(f"参数解析有误：{re_llm_res.group(2)}")

# 查询股票的 K 线图（日K，周K，月K）
import streamlit as st, requests, pandas as pd
from datetime import datetime, date, timedelta
import plotly.graph_objects as po

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

PATH_URL = "http://127.0.0.1:8000/stock"
TYPE = {
    "日K": "/get_stock_day_kline",
    "周K": "/get_stock_week_kline",
    "月K": "/get_stock_month_kline",
}

# 查询条件
with st.form("K线图"):
    col1, col2 = st.columns(2)
    with col1:
        stock_code = st.text_input(
            label="股票代码",
            value="sh600938"
        )

        type_k = st.selectbox(
            label="K线图类型",
            options=TYPE.keys(),
            index=0
        )

    with col2:
        start_date = st.date_input(
            label="开始时间",
            value=date.today() - timedelta(days=90)
        )
        end_date = st.date_input(
            label="结束时间",
            value=date.today()
        )

    sub_but = st.form_submit_button(label="绘制K线图")


# 获取 K线图信息
def get_k_data():
    url = f"{PATH_URL}{TYPE.get(type_k)}"
    data = {
        "code": stock_code,
        "startDate": str(start_date),
        "endDate": str(end_date),
        "type": 0,
    }

    try:
        res = requests.get(url, params=data)

        # 检查响应状态码
        res.raise_for_status()

        res_json = res.json()

        # 将 res_json["data"] 转换为 DataFrame
        data_pd = pd.DataFrame(res_json["data"])
        # 保留前六列
        data_pd = data_pd.iloc[:, :6]
        # 替换列名
        data_pd.columns = [
            "Date", "Close", "Open", "High", "Low", "Volume"
        ]

        # 列值判断
        data_pd["Date"] = pd.to_datetime(data_pd["Date"])
        for col in ["Close", "Open", "High", "Low", "Volume"]:
            data_pd[col] = pd.to_numeric(data_pd[col], "coerce")

        return res_json, data_pd

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到后端服务 ({url})。请确保后端服务正在运行。")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

    return []


# 绘制K线图
def draw_k(k_data_pd):
    # 使用 plotly.graph_objects 库
    # Candlestick 专门绘制金融K线图
    st.markdown(f"### {type_k}线图")
    k_fig = po.Figure(data=[po.Candlestick(
        x=k_data_pd['Date'],
        open=k_data_pd['Open'],
        high=k_data_pd['High'],
        low=k_data_pd['Low'],
        close=k_data_pd['Close'],
        name='股票K线',
        increasing_line_color='green',  # 可简写（旧版兼容）
        decreasing_line_color='red'
    )])

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

    # 绘制 柱状图
    st.markdown(f"### {type_k}交易量柱状图")
    b_fig = po.Figure(data=[po.Bar(
        x=k_data_pd['Date'],
        y=k_data_pd['Volume'],
        name='交易量',
        marker_color='steelblue',
    )])

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


if sub_but:
    with st.spinner("正在绘制K线图..."):
        # 获取 K线图信息
        k_data, k_data_pd = get_k_data()

        if k_data_pd.empty:
            st.error(f"未获取到 {stock_code} 股票信息")
        else:
            # 绘制 K线图
            draw_k(k_data_pd)

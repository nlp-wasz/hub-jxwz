# 查看股票大盘信息
import streamlit as st, requests, pandas as pd

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")


# 获取大盘数据（缓存）
@st.cache_data(ttl=3600, show_spinner="正在获取股票大盘数据")
def get_stock_board():
    url = ""
    try:
        url = "http://127.0.0.1:8000/stock/get_stock_board_info"
        res = requests.get(url)

        # 检查响应状态码
        res.raise_for_status()

        res_json = res.json()

        return res_json["data"]

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到后端服务 ({url})。请确保后端服务正在运行。")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

    return []


# 对指定列的信息 采用特殊style风格
def col_style(val):
    try:
        # 转换为 float
        col_value = float(val)

        if col_value > 0:
            return "color:green"
        else:
            return "color:red"
    except Exception as e:
        return 'color:black'


with st.spinner("正在查询"):
    # 获取 大盘数据
    board_data = get_stock_board()

    if not board_data:
        st.warning(f"未检索到相关数据！")
    else:
        # 转换为 DataFarme
        board_pd = pd.DataFrame(board_data)

        # 去除掉一些无用 字段信息
        board_pd = board_pd.loc[:, ['name', 'code', 'price', 'priceChange', 'changePercent',
                                    'open', 'high', 'low', 'volume', 'turnover', 'date']]

        board_pd.columns = [
            '指数名称', '代码', '最新价', '涨跌额', '涨跌幅(%)',
            '今开', '最高', '最低', '成交量', '成交额(万)', '更新时间'
        ]

        st.markdown("### 核心指数")
        # 遍历 board_pd，将每一个大盘数据 使用 st.metric 展示
        # 每行展示3列
        cols = st.columns(3)
        for index, row in board_pd.iterrows():
            col = cols[index % 3]

            name = row["指数名称"]
            price = row["最新价"]
            priceChange = row["涨跌额"]
            changePercent = row["涨跌幅(%)"]

            with col:
                st.metric(
                    label=name,
                    value=price,
                    delta=f"{priceChange} ({changePercent}%)",
                    delta_color="normal"
                )

        st.markdown("### 详细交易数据")
        # 将 board_pd 数据使用 st.dataframe() 展示
        # board_pd.style.map() 对指定列的信息 采用特殊style风格
        st.dataframe(
            data=board_pd.style.map(col_style, subset=['涨跌额', '涨跌幅(%)']),
            hide_index=True
        )

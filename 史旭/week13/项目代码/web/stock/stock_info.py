# 查看股票详细信息

import streamlit as st, requests, pandas as pd
from datetime import datetime, timedelta

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")


# 查询股票 详细信息
def get_stock_info(stock_code: str):
    url = "http://127.0.0.1:8000/stock/get_stock_info"
    data = {
        "code": stock_code
    }

    try:
        res = requests.get(url, params=data)

        # 检查响应状态码
        res.raise_for_status()

        res_json = res.json()

        return res_json["data"][0]

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到后端服务 ({url})。请确保后端服务正在运行。")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

    return None


# 展示 查询到的信息
def show_stock_info(stock_info):
    # 企业 和 代码
    st.subheader(f"📈 {stock_info.get('name', 'N/A')} ({stock_info.get('code', 'N/A')})")

    # 更新时间
    up_date = datetime.strptime(stock_info['date'], '%Y-%m-%d %H:%M:%S')
    st.caption(f"最新更新时间：{up_date.year}年{up_date.month}月{up_date.day}日 {up_date.strftime('%H-%M-%S')}")

    # 涨幅情况
    st.metric(label="涨跌幅度", value=stock_info["price"],
              delta=f"{stock_info["priceChange"]} ({stock_info["changePercent"]}%)", delta_color="normal")

    st.markdown("---")

    st.subheader("交易细节")
    # 使用 DataFrame 或 metric 进行指标展示
    metrics_data = {
        "今开": stock_info.get('open', 'N/A'),
        "昨收": stock_info.get('close', 'N/A'),
        "最高": stock_info.get('high', 'N/A'),
        "最低": stock_info.get('low', 'N/A'),
        "成交量 (手)": stock_info.get('volume', 'N/A'),
        "成交额 (万)": stock_info.get('turnover', 'N/A'),
        "换手率 (%)": stock_info.get('turnoverRate', 'N/A'),
        "量比": stock_info.get('volumeRate', 'N/A'),
    }
    metrics_data_pd = pd.DataFrame(list(metrics_data.items()), columns=["指标", "值"])
    st.dataframe(data=metrics_data_pd, hide_index=True)

    st.markdown("---")

    st.subheader("财务与估值")
    col_pe, col_spe, col_pb, col_worth = st.columns(4)

    col_pe.metric("市盈率(PE)", stock_info.get('pe', 'N/A'))
    col_spe.metric("静态市盈率(SPE)", stock_info.get('spe', 'N/A'))
    col_pb.metric("市净率(PB)", stock_info.get('pb', 'N/A'))
    col_worth.metric("总市值(亿)", stock_info.get('totalWorth', 'N/A'))

    st.markdown("---")

    st.subheader("买卖盘口")
    # 获取买卖盘口信息
    buy_data = stock_info["buy"]
    sell_data = stock_info["sell"]

    buy_list = []
    sell_list = []
    for i in range(0, len(buy_data), 2):
        # 获取第 i 手，买的价格
        buy_list.append([f"买{i // 2 + 1}", buy_data[i], buy_data[i + 1]])

    for i in range(0, len(sell_data), 2):
        # 获取第 i 手，卖的价格
        sell_list.append([f"卖{len(sell_data) // 2 - i // 2}", sell_data[i], sell_data[i + 1]])

    # sell_data 翻转
    sell_list.reverse()

    # 使用 st.dataframe 展示
    buy_list_pd = pd.DataFrame(buy_list, columns=['档位', '价格', '数量(手)'])
    sell_list_pd = pd.DataFrame(sell_list, columns=['档位', '价格', '数量(手)'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 买")
        st.dataframe(buy_list_pd, hide_index=True)
    with col2:
        st.markdown("#### 卖")
        st.dataframe(sell_list_pd, hide_index=True)

    st.markdown("---")

    st.subheader("详细信息")
    st.json(stock_info)


# 查询条件
with st.form("查询股票详细信息"):
    stock_code = st.text_input(
        label="股票代码",
        placeholder="请输入要查询的股票代码",
        value="sz002392"
    )
    sum_but = st.form_submit_button("查询")

    if sum_but:
        if not stock_code:
            st.warning("请输入要查询的股票代码")
        else:
            with st.spinner("正在查询..."):
                # 调用后端 API
                stock_info = get_stock_info(stock_code)

            if stock_info is None:
                st.error(f"未查询到 {stock_code} 股票信息！")
            else:
                # 展示 查询到的信息
                show_stock_info(stock_info)

                # st.json(stock_info)

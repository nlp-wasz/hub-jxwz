# 股票排名 页面
import pandas as pd
import requests
import streamlit as st

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

NODE_OPTIONS = {
    "沪深A股 (a)": 'a',
    "上交所A股 (ash)": 'ash',
    "深交所A股 (asz)": 'asz',
    "上交所B股 (bsh)": 'bsh',
    "深交所B股 (bsz)": 'bsz',
}

SORT_OPTIONS = {
    "交易价格": 'price',
    "涨跌额": 'priceChange',
    "涨跌幅": 'changePercent',
    "成交量": 'volume',
    "成交额": 'turnover',
    "今开盘": 'open',
    "最高价": 'high',
    "最低价": 'low',
    # 更多字段可以根据需要添加...
}


# 查询 股票排名 信息
@st.cache_data(ttl=3600, show_spinner="正在查询股票排名数据...")
def get_stock_rank(node: str,
                   industry_code: str = None,
                   page_index: int = 1,
                   page_size: int = 100,
                   sort_field: str = "price",
                   asc: int = 0):
    # 调用 API
    url = ""
    try:
        # 调用 股票检索API
        url = "http://127.0.0.1:8000/stock/get_stock_rank"
        header = {}
        data = {
            "node": node,
            "industryCode": industry_code,
            "pageIndex": page_index,
            "pageSize": page_size,
            "sort": sort_field,
            "asc": asc
        }

        res = requests.get(url, params=data)
        print(f"data:{data}")

        # 检查状态码是否正常（抛出 HTTPError 异常）
        res.raise_for_status()

        # 状态码正常，则将res转换为json
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


# 查询条件
st.markdown("### 检索条件")
st_node = st.selectbox(label="股票市场/板块代码", options=NODE_OPTIONS.keys(), index=0)
st_industryCode = st.selectbox(label="行业代码", options=["sw_sysh"], index=0)
st_pageSize = st.number_input(label="每页数量", min_value=10, max_value=100, step=10, value=20)
st_sort = st.selectbox(label="排序字段", options=SORT_OPTIONS.keys(), index=0)
st_asc = st.selectbox(label="排序方式", options=["降序", "升序"], index=0)

if st.button("查询"):
    st.session_state["pageIndex"] = 1

with st.spinner("正在获取数据..."):
    # 获取 股票排名信息
    node = NODE_OPTIONS[st_node]
    industryCode = st_industryCode
    pageIndex = st.session_state.get("pageIndex", 1)
    pageSize = st_pageSize
    sort = SORT_OPTIONS[st_sort]
    asc = 0 if st_asc == "降序" else 1

    print(f"pageIndex1:{pageIndex}")

    # 查询结果
    stock_rank_res = get_stock_rank(node, industryCode, pageIndex, pageSize, sort, asc)

    st.markdown("### 股票排名")
    if not stock_rank_res:
        st.warning(f"未检索到相关数据！")
    else:
        # 获取 主要数据信息
        rank_data = stock_rank_res["rank"]
        totalRecord = stock_rank_res["totalRecord"]

        # 将 rank_data 转换为 pd.DataFrame()
        rank_data_pd = pd.DataFrame(rank_data)
        rank_data_pd = rank_data_pd.loc[:, ['name', 'code', 'price', 'priceChange', 'changePercent',
                                            'open', 'high', 'low', 'volume', 'turnover', 'date']]
        rank_data_pd.columns = [
            '指数名称', '代码', '最新价', '涨跌额', '涨跌幅(%)',
            '今开', '最高', '最低', '成交量', '成交额(万)', '更新时间'
        ]

        # 展示数据量
        # 分页组件
        # 计算 总页数
        pages = totalRecord / pageSize if totalRecord % pageSize == 0 else totalRecord // pageSize + 1
        st.info(
            f"总计查询 {totalRecord} 条数据，目前展示 {pageSize if pageIndex < pages else totalRecord - (pageIndex - 1) * pageSize} 条数据，当前页码 {pageIndex}")

        # st.dataframe 展示
        st.dataframe(
            data=rank_data_pd.style.map(col_style, subset=['涨跌额', '涨跌幅(%)']),
            hide_index=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if pageIndex > 1 and st.button("上一页"):
                st.session_state["pageIndex"] = pageIndex - 1

        with col2:
            st.markdown(f"##### 当前第 {pageIndex} 页")

        with col3:
            if pageIndex < pages and st.button("下一页"):
                st.session_state["pageIndex"] = pageIndex + 1

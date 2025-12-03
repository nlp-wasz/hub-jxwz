# 查询 股票板块 信息
import streamlit as st, requests, pandas as pd

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")


# 定义 streamlit 缓存函数，对查询到的数据进行缓存
@st.cache_data(ttl=3600, show_spinner="正在查询股票板块数据...")
def get_stock_industry():
    url = ""
    try:
        # 直接调用 股票板块 API
        url = "http://127.0.0.1:8000/stock/get_stock_industry_code"
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


with st.spinner("正在查询："):
    # 获取 查询结果
    industry_data = get_stock_industry()

    if not industry_data:
        st.warning(f"未检索到相关数据！")
    else:
        # 转换为 DataFrame
        industry_data_pd = pd.DataFrame(data=industry_data)
        industry_data_pd.rename(columns={"industryCode": "板块代码", "name": "板块名称"}, inplace=True)
        industry_data_pd.index = pd.RangeIndex(start=1, stop=len(industry_data) + 1)

        st.dataframe(data=industry_data_pd)

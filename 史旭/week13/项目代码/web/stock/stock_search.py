# 股票搜索页面（搜索股票代码）

import streamlit as st, requests, pandas as pd

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

with st.form("Stock Search"):
    # 股票 检索框
    stock_like = st.text_input(
        label="根据股票名称或代码实现模糊搜索",
        placeholder="请输入关键字"
    )

    sub_but = st.form_submit_button("搜索")

    if sub_but and stock_like:
        with st.spinner(f"正在检索关键字 **{stock_like}**  ..."):
            try:
                # 调用 股票检索API
                url = "http://127.0.0.1:8000/stock/get_all_stock_code"
                header = {}
                data = {
                    "keyword": stock_like
                }

                res = requests.get(url, params=data)

                # 检查状态码是否正常（抛出 HTTPError 异常）
                res.raise_for_status()

                # 状态码正常，则将res转换为json
                res_json = res.json()

                # 使用 st.dataform() 展示
                res_data = pd.DataFrame(data=res_json["data"], columns=["代码", "企业"])
                res_data.index = pd.RangeIndex(start=1, stop=len(res_data) + 1)

                st.dataframe(res_data)

            except requests.exceptions.ConnectionError as e:
                st.error(f"连接错误：无法连接到后端服务 ({url})。请确保后端服务正在运行。")
            except requests.exceptions.HTTPError as e:
                st.error(f"API 请求失败：{e}")
            except Exception as e:
                st.error(f"未知错误：{e}")
    elif sub_but and not stock_like:
        st.warning("请输入要查询的关键字！")

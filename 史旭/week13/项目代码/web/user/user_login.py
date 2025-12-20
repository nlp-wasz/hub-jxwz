# 用户登录界面
import time

import streamlit as st, requests

with st.form("用户登录"):
    user_name = st.text_input(label="账号名称", placeholder="请输入账号名称")
    user_pass = st.text_input(label="密码", type="password")

    if st.form_submit_button("登录"):
        with st.spinner("正在登录..."):
            if user_name is None or user_pass is None:
                st.error("请填写完整信息")
            else:
                # 调用 用户登录API
                url = "http://127.0.0.1:8000/v1/user/user_login"
                header = {}
                data = {
                    "user_name": user_name,
                    "user_pass": user_pass,
                    "user_role": ""
                }

                res = requests.post(url, headers=header, json=data).json()

                if res["res_code"] == 200:
                    st.success(res["res_mess"])

                    # 在 st.session_state 中缓存登录信息
                    st.session_state["is_login"] = True
                    st.session_state["login_user_name"] = user_name

                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"res_mess：{res["res_mess"]}\n\nres_error：{res["res_error"]}")
                    # 在 st.session_state 中缓存登录信息
                    st.session_state["is_login"] = False
                    st.session_state["login_user_name"] = None

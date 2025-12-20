# 删除用户信息（用户账号注销）
import time

import requests
import streamlit as st

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

if st.button("是否注销当前账号！"):
    with st.spinner("正在注销..."):
        url = "http://127.0.0.1:8000/v1/user/deleteUserByUserName"
        header = {}
        data = {
            "user_name": st.session_state.get("login_user_name", None),
            "user_pass": "",
            "user_role": "",
        }

        res = requests.post(url, headers=header, json=data).json()
        if res["res_code"] == 200:
            st.success(res["res_result"])

            # 删除缓存信息
            st.session_state["is_login"] = False
            st.session_state["login_user_name"] = None
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"res_mess：{res["res_result"]}\n\nres_error：{res["res_error"]}")

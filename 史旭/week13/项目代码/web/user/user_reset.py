# 个人信息 修改界面
import time

import requests, streamlit as st

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

with st.spinner("正在获取个人信息..."):
    url = "http://127.0.0.1:8000/v1/user/byUserNameGetInfo"
    header = {}
    data = {
        "user_name": st.session_state.get("login_user_name", None)
    }

    res = requests.get(url, headers=header, params=data).json()

    if res["res_code"] == 200:
        st.dataframe(res["res_result"])

        # 展示能够修改的信息
        with st.form("修改个人信息"):
            user_name = st.text_input(label="用户名", value=res["res_result"]["user_name"])
            user_pass = st.text_input(label="用户名", value=res["res_result"]["user_password"], type="password")

            if st.form_submit_button("修改"):
                if not user_name or not user_pass:
                    st.error("请填写完整信息")

                # 调用用户信息修改 API
                url = "http://127.0.0.1:8000/v1/user/byUserNameUpdateInfo"
                header = {}
                data = {
                    "user_name": st.session_state.get("login_user_name", None),
                    "update_user_name": user_name,
                    "user_pass": user_pass,
                }

                res = requests.post(url, headers=header, data=data).json()

                if res["res_code"] == 200:
                    st.success(res["res_result"])

                    # 更新 session 缓存
                    st.session_state["login_user_name"] = user_name
                    time.sleep(1)
                    st.rerun()

                else:
                    st.error(f"{res["res_result"]}\n\n{res["res_error"]}")
    else:
        st.error(f"r{res["res_mess"]}\n\n{res["res_error"]}")

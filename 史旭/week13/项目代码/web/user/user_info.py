# 个人信息界面
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
    else:
        st.error(f"res_mess：{res["res_mess"]}\n\nres_error：{res["res_error"]}")

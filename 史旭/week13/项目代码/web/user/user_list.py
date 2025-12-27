import streamlit as st,requests

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")

with st.spinner("正在获取权限下的用户信息"):
    url = "http://127.0.0.1:8000/v1/user/byRoleGetUserInfo"
    header = {}
    data = {
        "user_name": st.session_state.get("login_user_name", None),
        "user_pass": "",
        "user_role": "",
    }

    res = requests.post(url, headers=header, json=data).json()

    st.dataframe(res["res_result"])

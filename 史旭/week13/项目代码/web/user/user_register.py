# 用户注册界面
import streamlit as st, requests

with st.form("用户信息注册"):
    user_name = st.text_input(label="账号名称", placeholder="请输入账号名称")
    user_pass = st.text_input(label="密码", type="password")
    user_role = st.selectbox(label="角色", options=["普通用户", "管理员"])

    if st.form_submit_button("注册"):
        with st.spinner("正在注册..."):
            if user_name is None or user_pass is None or user_role is None:
                st.error("请填写完整信息")
            else:
                # 调用 用户注册API
                url = "http://127.0.0.1:8000/v1/user/user_register"
                header = {}
                data = {
                    "user_name": user_name,
                    "user_pass": user_pass,
                    "user_role": user_role
                }

                res = requests.post(url, headers=header, json=data).json()

                if res["res_code"] == 200:
                    st.success(res["res_mess"])
                else:
                    st.error(f"res_mess：{res["res_mess"]}\n\nres_error：{res["res_error"]}")

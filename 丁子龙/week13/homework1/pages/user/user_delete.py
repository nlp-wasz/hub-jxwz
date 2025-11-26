import streamlit as st
import time, requests

def delete_user(user_name):
    response = requests.post(
        "http://127.0.0.1:8000/v1/users/delete",
        params={"user_name": user_name}
    ).json()

    st.write(response)

    if response['code'] == 200:
        return True
    else:
        return False

def user_login_page():
    # 检查是否已登录
    if st.session_state.get('logged', False):
        st.info(f"您已登录为 **{st.session_state['user_name']}**。")

        # 退出按钮
        if st.button("删除用户"):
            delete_user(st.session_state['user_name'])
            st.session_state['logged'] = False
            st.session_state['user_name'] = None
            time.sleep(0.5)
            st.rerun()  # 重新运行页面以显示未登录状态
        return


if __name__ == '__main__':
    user_login_page()
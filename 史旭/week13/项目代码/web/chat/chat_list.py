# 聊天历史  展示界面
import time

import streamlit as st, requests, pandas as pd

st.info(f"您已登录为 **{st.session_state['login_user_name']}**。")


# 获取 所有的聊天历史信息
def get_session():
    url = 'http://127.0.0.1:8000/v1/chat/get_chat_list'
    data = {
        "user_name": st.session_state['login_user_name'],
    }

    try:
        res = requests.get(url, params=data)

        # 校验 状态码
        res.raise_for_status()

        if res.json()["res_code"] == 200:
            return res.json()["res_result"]

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接错误：无法连接到 {url} 服务端！")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
    except Exception as e:
        st.error(f"未知错误：{e}")

    return None


with st.spinner("正在获取历史聊天记录..."):
    chat_session = get_session()

    if chat_session is None:
        st.error("为获取到聊天记录")

    # 循环遍历
    for chat in chat_session:
        # 展示信息
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 2])

            with col1:
                st.markdown(f"**{chat['session_id']}**")
                st.markdown(f"**{chat['chat_title']}**")

                st.caption(chat['created_at'])

        # 反馈
        with col2:
            st.markdown(f"**反馈**")
            st.write(chat["chat_feedback"])

        # 操作按钮
        with col3:
            if st.button("进入聊天界面", key=f"{chat["session_id"]}_switch"):
                # 跳转到 chat 聊天界面
                st.session_state["session_id"] = chat["session_id"]
                st.session_state["message"] = []
                st.session_state["is_load_session_message"] = True

                st.switch_page("chat/chat.py")

            if st.button("删除聊天记录", key=f"{chat["session_id"]}_delete"):
                url = "http://127.0.0.1:8000/v1/chat/delete_chat_message"
                data = {
                    "session_id": chat["session_id"],
                }
                try:
                    res = requests.get(url, params=data)

                    # 校验 状态码
                    res.raise_for_status()

                    st.session_state["session_id"] = None
                    st.session_state["message"] = []

                except requests.exceptions.ConnectionError as e:
                    st.error(f"连接错误：无法连接到 {url} 服务端！")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API 请求失败：{e}")
                except Exception as e:
                    st.error(f"未知错误：{e}")

                st.rerun()

        st.divider()

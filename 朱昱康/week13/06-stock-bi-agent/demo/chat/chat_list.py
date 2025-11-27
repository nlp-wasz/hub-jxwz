import streamlit as st
import requests

if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")

    data = requests.post("http://127.0.0.1:8000/v1/chat/list?user_name=" + st.session_state['user_name'])
    chat_data = data.json()["data"][::-1]

    # 为每个聊天会话创建卡片式展示
    for chat in chat_data:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"**{chat["session_id"]} / {chat['title']}**")
                st.caption(f"创建时间: {chat['start_time']}")

            with col2:
                feedback_text = "暂无反馈" if chat['feedback'] is None else chat['feedback']
                st.text(f"反馈: {feedback_text}")

            with col3:
                # 使用HTML a标签实现页面内跳转
                session_id = chat['session_id']
                st.session_state.session_id = session_id
                if st.button("进入聊天", key=session_id + "chat"):
                    st.switch_page("chat/chat.py")

                if st.button("删除聊天", key=session_id + "del"):
                    requests.post("http://127.0.0.1:8000/v1/chat/delete?session_id=" + session_id)

                    st.rerun()



            st.divider()

else:
    st.info("请先登录再使用模型～")
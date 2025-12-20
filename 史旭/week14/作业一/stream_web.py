import streamlit as st
from rag_llm import llm_qa

st.header("建模问答")

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "欢迎使用建模问答AI智能助手"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("建模问答"):
    # 调用 rag_llm.py 问答方法
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("正在回答："):
        llm_flush = st.empty()
        llm_res = ""

        for chunk in llm_qa(prompt):
            llm_res += chunk
            llm_flush.write(llm_res + " ▌")

        llm_flush.write(llm_res)

    st.session_state["messages"].append({
        "role": "user",
        "content": llm_res
    })

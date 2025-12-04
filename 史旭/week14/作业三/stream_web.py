import asyncio

import streamlit as st
from mcp_rag import agent_llm

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
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        llm_flush = st.empty()
        llm_res = ""


        async def consume():
            global llm_res

            async for chunk in agent_llm(prompt):
                llm_res += chunk
                llm_flush.markdown(llm_res + " ▌")


        # 异步调用
        with st.spinner("正在回答..."):
            asyncio.run(consume())
            llm_flush.markdown(llm_res)

st.session_state["messages"].append({
    "role": "assistant",
    "content": llm_res
})

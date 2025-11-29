import streamlit as st
import time, requests

def get_user(user_name):
    response = requests.post(
        "http://127.0.0.1:8000/v1/users/info",
        params={"user_name": user_name}
    ).json()

    st.write(response)

    if response['code'] == 200:
        return True
    else:
        return False

if st.session_state.get('logged', False):
    get_user(st.session_state['user_name'])

# 用户信息管理  业务逻辑 模块（用户信息 CRUD）
import asyncio
import json

import httpx
import pandas as pd
from httpx import AsyncClient

from sqlalchemy import and_
from model_type.SqliteOrm import SessionLocal, UserTable
from model_type.RequestResponse import UserLoginRequest, ChatRequest

# from model_type.SqliteOrm import SessionLocal, UserTable
# from model_type.RequestResponse import UserLoginRequest
from servers import ChatServer


# 用户登录
def user_login(use: UserLoginRequest):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(and_(UserTable.user_name == use.user_name, UserTable.user_password == use.user_pass)) \
            .first()

        if user_info:
            return True
        else:
            return False


# 用户注册
def user_register(use: UserLoginRequest):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == use.user_name) \
            .first()

        if user_info:
            # 用户信息 已存在
            return False
        else:
            try:
                # 添加注册账号信息
                user_add = UserTable(
                    user_name=use.user_name,
                    user_password=use.user_pass,
                    user_role=use.user_role,
                    user_status=True
                )
                session.add(user_add)
                session.commit()

                return True
            except Exception as e:
                session.rollback()

                return False


# 根据用户名 获取用户信息
def byUserNameGetInfo(user_name: str):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == user_name) \
            .first()

        print(user_info)
        if user_info:
            # 用户信息存在
            return True
        else:
            return False


# 个人信息界面
import requests, streamlit as st


def req():
    url = "http://127.0.0.1:8000/stock/get_stock_board_info"
    header = {}

    res = requests.get(url).json()
    res_pd = pd.DataFrame(res["data"])

    for i, row in res_pd.iterrows():
        print(i)
        print(row)


# 发送聊天请求（后端LLM生成回答）
async def chat_request(prompt):
    # 发送 异步 Http请求
    url = "http://127.0.0.1:8000/v1/chat/"
    header = {}
    data = {
        "prompt": prompt
    }

    async with AsyncClient(timeout=httpx.Timeout(60)) as client:
        async with client.stream("POST", url, headers=header, json=data) as response:
            async for chunk in response.aiter_text():
                if chunk:
                    yield chunk


# 发送http请求，调用chat API，流式输出（需要使用异步函数接收）
async def consume_stream():
    chat_request = ChatRequest(prompt="请帮我查询北京的天气状况", user_name="xiaoxu",
                               session_id="853b7e90900b4af49f348f086065ae28", select_tools=["get_city_weather"])
    res = ChatServer.chat(chat_request)
    async for chunk in res:
        print(chunk, end="")

    # async for chunk in chat_request("你好"):
    #     print(chunk)


# 解析 json字符串
def load_json():
    import re
    text = """
        ```json
        get_stock_month_kline:{"code": "sh600938", "startDate": "2025-08-30", "endDate": "2025-11-28", "type": 0}
        ```
        查询到的股票代码 sh600938 从 2025-08-30 到 2025-11-28 的月K线数据如下：
        
        - 2025-09-30: 开盘价 25.69, 最高价 26.90, 最低价 25.41, 收盘价 26.13, 成交量 9776357.000
        - 2025-10-31: 开盘价 26.01, 最高价 27.95, 最低价 25.50, 收盘价 27.11, 成交量 8016820.000
        - 2025-11-28: 开盘价 27.14, 最高价 30.07, 最低价 27.14, 收盘价 27.79, 成交量 7388710.000
        
        请注意，这些是预测数据，并不代表实际市场表现。
        
        使用re，从上面字符串中解析json字符串包裹并使用json.load加载为python对象
    """
    pattern = r'(\w+):({.*?})'
    a = re.search(pattern, text)

    if a:
        print("a")


if __name__ == '__main__':
    # byUserNameGetInfo("小旭")
    # 异步执行  consume_stream
    # asyncio.run(consume_stream())

    load_json()

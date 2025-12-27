# streamlit 主界面 模块

import streamlit as st

# --- 账户中心 ---
page_user_register = st.Page("user/user_register.py", title="用户注册", icon="➕")
page_user_login = st.Page("user/user_login.py", title="登陆/退出", icon="🚪")
page_user_info = st.Page("user/user_info.py", title="个人信息", icon="👤")
page_user_reset = st.Page("user/user_reset.py", title="修改信息", icon="✏️")
page_user_delete = st.Page("user/user_delete.py", title="删除账户", icon="❌")
page_user_list = st.Page("user/user_list.py", title="列举账户", icon="👥")

# --- 股票中心 ---
page_stock_search = st.Page("stock/stock_search.py", title="股票搜索", icon="🔍")
page_stock_industry = st.Page("stock/stock_industry.py", title="股票板块", icon="🏷️")
page_stock_board = st.Page("stock/stock_board.py", title="股票大盘", icon="🧩")
page_stock_rank = st.Page("stock/stock_rank.py", title="股票排行", icon="🏆")
page_stock_info = st.Page("stock/stock_info.py", title="股票信息", icon="ℹ️")
page_stock_kline = st.Page("stock/stock_kline.py", title="股票K线图", icon="📊")
page_stock_min = st.Page("stock/stock_min_data.py", title="当日交易", icon="📊")
page_stock_fav = st.Page("stock/stock_favorite.py", title="股票收藏", icon="⭐")

# --- 聊天中心 ---
page_chat = st.Page("chat/chat.py", title="通用对话", icon="💬")
page_chat_list = st.Page("chat/chat_list.py", title="对话历史", icon="🕰️")

# 判断用户是否已经登录
if st.session_state.get("is_login", False):
    na = st.navigation(
        {
            "账户中心": [page_user_login, page_user_register, page_user_info, page_user_reset, page_user_delete,
                         page_user_list],
            "股票中心": [page_stock_search, page_stock_industry, page_stock_board, page_stock_rank, page_stock_info,
                         page_stock_kline, page_stock_min, page_stock_fav],
            "聊天中心": [page_chat, page_chat_list],
        }
    )
else:
    na = st.navigation(
        {
            "用户信息": [page_user_login, page_user_register]
        },
        position="sidebar"
    )

# 导航栏
na.run()

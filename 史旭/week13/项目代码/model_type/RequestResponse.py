# 请求和响应体  结构
from datetime import datetime, date
from typing import Union, Annotated, Any, Optional, List, Dict
from pydantic import BaseModel, Field


# 通用 请求体
class PublicRequest(BaseModel):
    """通用请求体"""
    request_id: int = Field(description="通用请求体ID")


class PublicResponse(BaseModel):
    """通用响应体"""
    res_code: int = Field(description="通用响应体CODE")
    res_result: Any = Field(description="通用响应体结果")
    res_mess: str = Field(description="通用响应体信息")
    res_error: str = Field(description="通用响应体错误代码")


# 用户 登录/注册 请求体
class UserLoginRequest(BaseModel):
    """用户登录请求体"""

    user_name: str = Field(description="用户名称")
    user_pass: str = Field(description="用户密码")
    user_role: str = Field(description="用户角色  管理员/普通用户")


# 用户信息 响应体
class UserInfoResponse(BaseModel):
    """用户信息 响应体"""
    user_id: int = Field(description="用户ID")
    user_name: str = Field(description="账号")
    user_password: str = Field(description="密码")
    user_role: str = Field(description="用户角色")
    user_status: bool = Field(description="用户状态")
    created_at: str = Field(description="注册时间")
    updated_at: str = Field(description="更新时间")


# 用户聊天请求体
class ChatRequest(BaseModel):
    """用户聊天请求体"""

    prompt: str = Field(description="请求的问题")
    user_name: str = Field(description="当前登录用户名")
    session_id: str = Field(description="当前聊天记录窗口缓存ID")
    select_tools: List[str] = Field(description="选择使用的MCP工具名称 列表")


# 用户聊天响应体
class ChatResponse(BaseModel):
    """用户聊天响应体"""
    generator: str = Field(description="LLM响应的内容")


# 用户聊天窗口 session 缓存响应体
class ChatSessionResponse(BaseModel):
    """用户聊天窗口 session 缓存响应体"""
    chat_id: int = Field(description="聊天窗口ID")
    session_id: str = Field(description="缓存ID")
    user_id: str = Field(description="用户名称（ID）")
    chat_title: str = Field(description="聊天窗口标题")
    chat_feedback: str = Field(description="用户反馈")
    feedback_at: str = Field(description="反馈时间")
    created_at: str = Field(description="创建时间")
    update_at: str = Field(description="更新时间")


# 用户聊天窗口 message 缓存响应体
class ChatMessageResponse(BaseModel):
    """用户聊天窗口 message 缓存响应体"""
    message_id: int = Field(description="聊天历史ID")
    chat_id: int = Field(description="聊天窗口ID")
    user_id: str = Field(description="用户名称（ID）")
    session_id: str = Field(description="缓存ID")

    message_role: str = Field(description="角色")
    message_content: str = Field(description="内容")
    message_feedback: str = Field(description="反馈")
    feedback_at: str = Field(description="反馈时间")
    created_at: str = Field(description="创建时间")
    update_at: str = Field(description="更新时间")

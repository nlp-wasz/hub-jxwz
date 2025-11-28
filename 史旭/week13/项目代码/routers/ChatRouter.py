# 聊天信息管理 API（Router 路由）
import traceback
from typing import Annotated

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

# from ..model_type.RequestResponse import ChatRequest, ChatResponse, PublicResponse
from model_type.RequestResponse import ChatRequest, ChatResponse, PublicResponse
from servers import ChatServer

chatRouter = APIRouter(prefix="/v1/chat", tags=["chat"])


@chatRouter.post("/")
async def chat(chat_request: ChatRequest) -> StreamingResponse:
    # 调用 server 层异步方法，获取LLM生成结果（这里使用异步函数包裹，也可以直接交给 StreamingResponse）
    async def async_llm_generator():
        async for chunk in ChatServer.chat(chat_request):
            if chunk:
                yield chunk

    # 流式输出
    return StreamingResponse(
        content=async_llm_generator(),  # 异步生成器
        media_type="text/event-stream"  # 流式输出
    )


# 生成 session_id
@chatRouter.get("/generator_session_id")
def generator_session_id():
    try:
        # 调用 chat 服务端
        session_id = ChatServer.generator_session_id()

        return PublicResponse(res_code=200, res_result={"session_id": session_id}, res_mess="", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=traceback.format_exc(), res_mess="", res_error="")


# 获取 缓存信息 列表（ChatSessionTable 表）
@chatRouter.get("/get_chat_list")
def get_chat_list(user_name: Annotated[str, "用户名（ID）"]):
    try:
        # 获取 缓存信息 列表（ChatSessionTable 表）
        chat_session = ChatServer.get_chat_list(user_name)

        return PublicResponse(res_code=200, res_result=chat_session, res_mess="", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=traceback.format_exc(), res_mess="", res_error="")


# 获取 缓存信息 列表（ChatMessageTable 表）
@chatRouter.get("/get_message_list")
def get_message_list(session_id: Annotated[str, "缓存ID"]):
    try:
        # 获取 缓存信息 列表（ChatSessionTable 表）
        chat_message = ChatServer.get_message_list(session_id)

        return PublicResponse(res_code=200, res_result=chat_message, res_mess="", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=traceback.format_exc(), res_mess="", res_error="")


# 根据 session_id 删除缓存信息
@chatRouter.get("/delete_chat_message")
def delete_chat_message(session_id: Annotated[str, "缓存ID"]):
    try:
        # 获取 缓存信息 列表（ChatSessionTable 表）
        is_delete = ChatServer.delete_chat_message(session_id)

        return PublicResponse(res_code=200, res_result=is_delete, res_mess="", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=traceback.format_exc(), res_mess="", res_error="")

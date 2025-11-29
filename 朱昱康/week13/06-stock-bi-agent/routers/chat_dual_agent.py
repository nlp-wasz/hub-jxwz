import traceback

from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.extensions.memory import AdvancedSQLiteSession
from fastapi import FastAPI, APIRouter  # type: ignore
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from typing import Union
import os  # Need to import os for environment variables

from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

import services.chat_dual_agent as chat_services
from models.data_models import BasicResponse, RequestForChat, ResponseForChat

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/")
async def chat(req: RequestForChat) -> StreamingResponse:
    try:
        # 异步函数
        async def chat_stream_generator():
            # async for 遍历异步数据流
            async for chunk in chat_services.chat(
                    user_name=req.user_name,
                    task=req.task,
                    session_id=req.session_id,
                    content=req.content,
                    tools=req.tools
            ):
                # 每次迭代获取一个数据块并立即 yield 返回
                yield chunk

        # Server-Sent Events (SSE) sse 对话流式输出，实时数据流
        return StreamingResponse(
            content=chat_stream_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/init")
async def init_chat() -> StreamingResponse:
    try:
        return BasicResponse(
            code=200, message="ok",
            data={
                "session_id": chat_services.generate_random_chat_id()
            }
        )
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/get")
def get_chat(session_id: str) -> BasicResponse:
    try:
        response = chat_services.get_chat_sessions(session_id)
        return BasicResponse(
            code=200, message="ok",
            data=response
        )
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/delete")
def delete_chat(session_id: str) -> BasicResponse:
    try:
        response = chat_services.delete_chat_session(session_id)
        return BasicResponse(code=200, message="ok", data=[])
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/list")
def list_chat(user_name: str) -> BasicResponse:
    try:
        chat_records = chat_services.list_chat(user_name)
        return BasicResponse(code=200, message="ok", data=chat_records)
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/feedback")
def feedback_chat(session_id: str, message_id: int, feedback: bool) -> BasicResponse:
    try:
        chat_services.change_message_feedback(session_id, message_id, feedback)
        return BasicResponse(code=200, message="ok", data=[])
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])


@router.post("/switch_agent")
async def switch_agent(session_id: str, agent_type: str) -> BasicResponse:
    """
    手动切换agent类型
    agent_type: "casual" 或 "analysis"
    """
    try:
        if agent_type not in ["casual", "analysis"]:
            return BasicResponse(code=400, message="Invalid agent type. Must be 'casual' or 'analysis'", data=[])
        
        # 这里可以实现手动切换agent的逻辑
        # 例如更新数据库中的agent类型状态
        
        return BasicResponse(
            code=200, 
            message=f"Successfully switched to {agent_type} agent", 
            data={"agent_type": agent_type}
        )
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=500, message=traceback.format_exc(), data=[])
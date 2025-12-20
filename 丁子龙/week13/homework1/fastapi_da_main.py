import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-ea07bf0880504b75a31b1bce38437fcf"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_MODEL"] = "qwen-max"
os.environ["OPENAI_VISON_MODEL"] = "qwen-vl"

import uvicorn
from fastapi import FastAPI  # type: ignore
from routers.user import router as user_routers
from routers.chat import router as chat_routers
from routers.data import router as data_routers
from routers.stock import router as stock_routers

from mcp_server.autostock import app as stock_app

app = FastAPI()


@app.get("/v1/healthy")
def read_healthy():
    pass

# 自定义的服务接口
app.include_router(user_routers)
app.include_router(chat_routers)
app.include_router(data_routers)
app.include_router(stock_routers)

app.mount("/stock", stock_app) # 底层stock api 接口

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn main_server:app

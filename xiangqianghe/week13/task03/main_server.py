import os
from dotenv import load_dotenv
load_dotenv(override=True)

import uvicorn
from fastapi import FastAPI  # type: ignore
from routers.user import router as user_routers
from routers.chat import router as chat_routers
from routers.data import router as data_routers
from routers.stock import router as stock_routers
from routers.mcp import router as mcp_routers

from api.autostock import app as stock_app

app = FastAPI()


@app.get("/v1/healthy")
def read_healthy():
    return {"status": "ok"}

# 自定义的服务接口
app.include_router(user_routers)
app.include_router(chat_routers)
app.include_router(data_routers)
app.include_router(stock_routers)
app.include_router(mcp_routers)

app.mount("/stock", stock_app) # 底层stock api 接口

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn main_server:app

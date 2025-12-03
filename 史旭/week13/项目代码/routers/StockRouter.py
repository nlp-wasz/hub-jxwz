# 股票信息查询 API （Router 路由）

from fastapi import APIRouter

stockRouter = APIRouter(prefix="/v1/stock", tags=["stock"])


@stockRouter.get("/")
async def index():
    pass

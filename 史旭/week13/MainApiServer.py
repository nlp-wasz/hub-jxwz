# FastAPI 主模块（用于加载所有 router 路由模块，以及 挂载 FastApi 服务）

import uvicorn, sys, pathlib

# 添加项目 根目录路径
# sys.path.append(str(pathlib.Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse

from fast_api.StockApiServer import stock_api
from routers.UserRouter import userRouter
from routers.StockRouter import stockRouter
from routers.ChatRouter import chatRouter

main_api = FastAPI(
    title="Stock API",
    description="用户股票管理 API",
    version="1.0.0",
    docs_url=None,
)


# fastapi页面swagger方式默认加载境外资源，这里修改为国内资源
@main_api.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API 文档",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


# 加载 Router 路由
main_api.include_router(userRouter)
main_api.include_router(stockRouter)
main_api.include_router(chatRouter)

# 挂载 FastAPI 服务（转发给 stock_api，但不会显示在docs接口文档）
main_api.mount("/stock", stock_api)

if __name__ == "__main__":
    uvicorn.run(main_api, host="127.0.0.1", port=8000, workers=1)

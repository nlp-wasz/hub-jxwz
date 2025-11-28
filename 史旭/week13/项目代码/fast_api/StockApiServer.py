# 股票信息查询 API
"""
股票信息查询网站
https://www.autostock.cn/#/trade/stock
https://s.apifox.cn/c3278b4f-5629-4732-858c-36758ff5d083/api-147275957
"""
import uvicorn

TOKEN = "zgaLG8unUPr"

import requests, traceback
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse
from typing import Union, Annotated, Any, Optional, Dict
from pydantic import BaseModel, Field

stock_api = FastAPI(
    title="StockInfo API",
    description="股票信息查询 API",
    version="1.0.0",
    docs_url=None,
)


# fastapi页面swagger方式默认加载境外资源，这里修改为国内资源
@stock_api.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API 文档",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


# path：http服务的路径
# operation_id：mcp服务的名字
# 获取所有股票代码，支持代码和名称模糊查询
@stock_api.get("/get_all_stock_code", operation_id="get_all_stock_code")
async def get_all_stock_code(
        keyword: Annotated[Optional[str], "支持代码和名称模糊查询"] = None
) -> Dict:
    """获取所有股票代码，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/all" + "?token=" + TOKEN
    if keyword:
        url += "&keyWord=" + keyword

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        print(F"get_all_stock_code：{url}")
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


# 获取所有指数，支持代码和名称模糊查询
@stock_api.get("/get_all_index_code", operation_id="get_all_index_code")
async def get_all_index_code():
    """获取所有指数，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/index/all" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


# 获取股票板块数据
@stock_api.get("/get_stock_industry_code", operation_id="get_stock_industry_code")
async def get_stock_industry_code():
    """获取股票板块数据"""
    url = "https://api.autostock.cn/v1/stock/industry/rank" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


# 获取股票大盘数据
@stock_api.get("/get_stock_board_info", operation_id="get_stock_board_info")
async def get_stock_board_info():
    """获取股票大盘数据"""
    url = "https://api.autostock.cn/v1/stock/board" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


# 股票排行
@stock_api.get("/get_stock_rank", operation_id="get_stock_rank")
async def get_stock_rank(
        node: Annotated[str, "股票市场/板块代码: {'a','b','ash','asz','bsh','bsz'} a(沪深A股)"],
        industryCode: Annotated[Optional[str], "行业代码，可选"] = None,
        pageIndex: Annotated[int, "页码"] = 1,
        pageSize: Annotated[int, "每页大小"] = 100,
        sort: Annotated[
            str, "排序字段: price,priceChange,pricePercent,buy,sell,open,close,high,low,volume,turnover 默认price(交易价格)。"] = "price",
        asc: Annotated[int, "排序方式: 0=降序(默认), 1=升序"] = 0
) -> Dict:
    """股票排行"""
    url = "https://api.autostock.cn/v1/stock/rank" + "?token=" + TOKEN
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        payload = {
            "node": node,
            "industryCode": industryCode,
            "pageIndex": pageIndex,
            "pageSize": pageSize,
            "sort": sort,
            "asc": asc
        }
        response = requests.request("POST", url, headers=headers, json=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


# 月k
@stock_api.get("/get_stock_month_kline", operation_id="get_stock_month_kline")
async def get_stock_month_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """月k"""
    url = "https://api.autostock.cn/v1/stock/kline/month" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


# 周k
@stock_api.get("/get_stock_week_kline", operation_id="get_stock_week_kline")
async def get_stock_week_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
):
    """周k"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


# 日k
@stock_api.get("/get_stock_day_kline", operation_id="get_stock_day_kline")
async def get_stock_day_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """日k"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


# 股票基础信息
@stock_api.get("/get_stock_info", operation_id="get_stock_info")
async def get_stock_info(code: Annotated[str, "股票代码"]) -> Dict:
    """股票基础信息"""
    url = "https://api.autostock.cn/v1/stock" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


# 分时信息
@stock_api.get("/get_stock_minute_data", operation_id="get_stock_minute_data")
async def get_stock_minute_data(code: str):
    """分时信息"""
    url = "https://api.autostock.cn/v1/stock/min" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


if __name__ == "__main__":
    uvicorn.run(stock_api, host="127.0.0.1", port=8001, workers=1)

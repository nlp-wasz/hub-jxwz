import traceback

from fastapi import FastAPI, APIRouter  # type: ignore
import services.stock as service_stock
from models.data_models import BasicResponse

router = APIRouter(prefix="/v1/stock", tags=["stocks"])


@router.post("/list_fav_stock")
def get_user_all_stock(user_name: str):
    try:
        return BasicResponse(code=200, message="获取用户所有股票成功", data=service_stock.get_user_all_stock(user_name))
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=404, message=traceback.format_exc(), data=[])

@router.post("/del_fav_stock")
def delete_user_stock(user_name: str, stock_code: str):
    try:
        return BasicResponse(code=200, message="删除成功", data=service_stock.delete_user_stock(user_name, stock_code))
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=404, message=traceback.format_exc(), data=[])

@router.post("/add_fav_stock")
def add_user_stock(user_name: str, stock_code: str):
    try:
        return BasicResponse(code=200, message="添加成功", data=service_stock.add_user_stock(user_name, stock_code))
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=404, message=traceback.format_exc(), data=[])

@router.post("/clear_fav_stock")
def clear_user_stock(user_name: str):
    try:
        return BasicResponse(code=200, message="删除成功", data=service_stock.clear_user_stock(user_name))
    except Exception as e:
        print(traceback.format_exc())
        return BasicResponse(code=404, message=traceback.format_exc(), data=[])



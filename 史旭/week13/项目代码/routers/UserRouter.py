# 用户信息管理 API（Router 路由）
import traceback
from fastapi import APIRouter, Query, Form

# from ..model_type.RequestResponse import UserLoginRequest, PublicResponse
# from ..servers import UserServer as userServer

from model_type.RequestResponse import UserLoginRequest, PublicResponse
from servers import UserServer as userServer

userRouter = APIRouter(prefix="/v1/user", tags=["user"])


# 用户登录
@userRouter.post("/user_login")
def user_login(user: UserLoginRequest):
    try:
        if userServer.user_login(user):
            return PublicResponse(res_code=200, res_result=True, res_mess="登录成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=False, res_mess="账号或密码输入错误", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="登录失败", res_error=traceback.format_exc())


# 用户注册
@userRouter.post("/user_register")
def user_register(user: UserLoginRequest):
    try:
        if userServer.user_register(user):
            return PublicResponse(res_code=200, res_result=True, res_mess="账号注册成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=False, res_mess="账号已存在", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="注册失败", res_error=traceback.format_exc())


# 根据用户名 获取用户信息
@userRouter.get("/byUserNameGetInfo")
def byUserNameGetInfo(user_name: str = Query(..., description="用户名")):
    try:
        if user_name is None:
            return PublicResponse(res_code=500, res_result=False, res_mess="未登录", res_error="")

        is_exists, user_info = userServer.byUserNameGetInfo(user_name)
        if is_exists:
            return PublicResponse(res_code=200, res_result=user_info, res_mess="查询成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=user_info, res_mess="查询失败", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="查询失败", res_error=traceback.format_exc())


# 根据用户名 修改用户信息（暂时只修改 用户名和密码）
@userRouter.post("/byUserNameUpdateInfo")
def byUserNameUpdateInfo(
        user_name: str = Form(..., description="用户名"),
        update_user_name: str = Form(..., description="修改后的用户名"),
        user_pass: str = Form(..., description="修改后的密码"),
):
    try:
        if not user_name or not update_user_name or not user_pass:
            return PublicResponse(res_code=500, res_result=False, res_mess="请填写完成信息", res_error="")

        is_update, message = userServer.byUserNameUpdateInfo(user_name, update_user_name, user_pass)
        if is_update:
            return PublicResponse(res_code=200, res_result=message, res_mess="修改成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=message, res_mess="修改失败", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="修改失败", res_error=traceback.format_exc())


# 根据用户名 删除用户信息
@userRouter.post("/deleteUserByUserName")
def deleteUserByUserName(user: UserLoginRequest):
    try:
        if not user.user_name:
            return PublicResponse(res_code=500, res_result=False, res_mess="请填写完成信息", res_error="")

        is_update, message = userServer.deleteUserByUserName(user.user_name)
        if is_update:
            return PublicResponse(res_code=200, res_result=message, res_mess="删除成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=message, res_mess="删除失败", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="删除失败", res_error=traceback.format_exc())


# 根据用户名的权限 查询权限下的所有用户信息
@userRouter.post("/byRoleGetUserInfo")
def byRoleGetUserInfo(user: UserLoginRequest):
    try:
        if not user.user_name:
            return PublicResponse(res_code=500, res_result=False, res_mess="请填写完成信息", res_error="")

        is_update, user_infos = userServer.byRoleGetUserInfo(user.user_name)
        if is_update:
            return PublicResponse(res_code=200, res_result=user_infos, res_mess="查询成功", res_error="")
        else:
            return PublicResponse(res_code=500, res_result=user_infos, res_mess="查询失败", res_error="")
    except Exception as e:
        return PublicResponse(res_code=500, res_result=False, res_mess="查询失败", res_error=traceback.format_exc())

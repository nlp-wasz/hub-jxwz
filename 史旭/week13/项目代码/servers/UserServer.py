# 用户信息管理  业务逻辑 模块（用户信息 CRUD）
from sqlalchemy import and_

# from ..model_type.SqliteOrm import SessionLocal, UserTable
# from ..model_type.RequestResponse import UserLoginRequest

from model_type.SqliteOrm import SessionLocal, UserTable
from model_type.RequestResponse import UserLoginRequest, UserInfoResponse


# 用户登录
def user_login(use: UserLoginRequest):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(and_(UserTable.user_name == use.user_name, UserTable.user_password == use.user_pass)) \
            .first()

        if user_info:
            return True
        else:
            return False


# 用户注册
def user_register(use: UserLoginRequest):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == use.user_name) \
            .first()

        if user_info:
            # 用户信息 已存在
            return False
        else:
            try:
                # 添加注册账号信息
                user_add = UserTable(
                    user_name=use.user_name,
                    user_password=use.user_pass,
                    user_role=use.user_role,
                    user_status=True
                )
                session.add(user_add)
                session.commit()

                return True
            except Exception as e:
                session.rollback()

                return False


# 根据用户名 获取用户信息
def byUserNameGetInfo(user_name: str):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == user_name) \
            .first()

        if user_info:
            # 用户信息存在
            return True, UserInfoResponse(
                user_id=user_info.user_id,
                user_name=user_info.user_name,
                user_password=user_info.user_password,
                user_role=user_info.user_role,
                user_status=user_info.user_status,
                created_at=user_info.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=user_info.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            return False, UserInfoResponse()


# 根据用户名 修改用户信息
def byUserNameUpdateInfo(user_name: str, update_user_name: str, user_pass: str):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == user_name) \
            .first()

        if user_info:
            try:
                # 用户信息存在
                user_info.user_name = update_user_name
                user_info.user_password = user_pass
                session.commit()

                return True, "用户信息修改完成！"
            except Exception as e:
                session.rollback()
                return False, "用户信息修改失败！"
        else:
            return False, "用户不存在！"


# 根据用户名 删除用户信息
def deleteUserByUserName(user_name: str):
    with SessionLocal() as session:
        try:
            # 用户查询
            session.query(UserTable) \
                .filter(UserTable.user_name == user_name) \
                .delete()
            session.commit()

            return True, "用户信息删除成功！"
        except Exception as e:
            session.rollback()
            return False, "用户信息删除失败！"


# 根据用户名的权限 查询权限下的所有用户信息
def byRoleGetUserInfo(user_name: str):
    with SessionLocal() as session:
        # 用户查询
        user_info = session.query(UserTable) \
            .filter(UserTable.user_name == user_name) \
            .first()

        infos = []
        infos.append(UserInfoResponse(
            user_id=user_info.user_id,
            user_name=user_info.user_name,
            user_password=user_info.user_password,
            user_role=user_info.user_role,
            user_status=user_info.user_status,
            created_at=user_info.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=user_info.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        ))

        if user_info:
            try:
                # 根据 user_role 字段，查询
                user_info_all = session.query(UserTable) \
                    .filter(UserTable.user_role == "普通用户") \
                    .all()

                infos.extend([UserInfoResponse(
                    user_id=i.user_id,
                    user_name=i.user_name,
                    user_password=i.user_password,
                    user_role=i.user_role,
                    user_status=i.user_status,
                    created_at=i.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at=i.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                ) for i in user_info_all])

                return True, infos
            except Exception as e:
                session.rollback()
                return False, []
        else:
            return False, "用户不存在！"

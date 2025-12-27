import hashlib
import traceback
from typing import Optional, List
from models.orm import UserTable, SessionLocal
from models.data_models import User

def password_hash(password: str) -> str:
    """对密码进行哈希处理。"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_user_exists(username: str) -> bool:
    """检查用户是否存在。"""
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == username).first()
            if user is None:
                return False

            return True
    except Exception as e:
        traceback.print_exc()
        return False

def user_register(user_name: str, password: str, user_role: str) -> bool:
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
        if user is not None:
            return False

        password = password_hash(password)
        user = UserTable(user_name=user_name, password=password, user_role=user_role, status=True)
        session.add(user)
        session.commit()
        return True

def get_user_info(user_name: str) -> Optional[User]:
    try:
        with SessionLocal() as session:
            user_db_record: UserTable|None = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user_db_record:
                return User(
                    user_id=user_db_record.id,
                    user_name=user_db_record.user_name,
                    user_role=user_db_record.user_role,
                    register_time=user_db_record.register_time,
                    status=user_db_record.status
                )
            else:
                return None
    except Exception as e:
        traceback.print_exc()
        return None

def list_users(page_index:int=1, page_size=200) -> List[User]:
    try:
        with SessionLocal() as session:
            user_db_records = session.query(UserTable).offset((page_index-1)*page_size).limit(page_size).all()
            return [
                User(
                user_id=user_db_record.id,
                user_name=user_db_record.user_name,
                user_role=user_db_record.user_role,
                register_time=user_db_record.register_time,
                status=user_db_record.status) for user_db_record in user_db_records
            ]
    except Exception as e:
        traceback.print_exc()
        return []

def user_login(username: str, password: str) -> bool:
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_name == username).first()
        if user is None:
            return False

        password = password_hash(password)
        if user.password != password:
            return False

        return True


def user_delete(user_name: str) -> bool:
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is None:
                return False

            session.delete(user)
            session.commit()
            return True
    except Exception as e:
        traceback.print_exc()
        return False

def user_reset_password(user_name: str, password: str) -> bool:
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
        if user is None:
            return False

        user.password = password_hash(password)  # type: ignore
        session.commit()
        return True

def alter_user_status(user_name: str, status: bool) -> bool:
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
        if user is None:
            return False
        user.status = status  # type: ignore
        session.commit()
        return True


def alter_user_role(user_name: str, user_role: str) -> bool:
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
        if user is None:
            return False

        user.user_role = user_role  # type: ignore
        session.commit()
        return True


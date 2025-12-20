import asyncio
from typing import Optional, List

from api.autostock import get_stock_info
from models.data_models import StockFavInfo
from models.orm import UserFavoriteStockTable, SessionLocal, UserTable

def get_user_all_stock(user_name: str) -> List[StockFavInfo]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return []
        else:
            user_id = user_id[0]
        user_stock_db_records = session.query(UserFavoriteStockTable).filter(UserFavoriteStockTable.user_id == user_id).all()
        return [
            StockFavInfo(
                stock_code=user_stock_db_record.stock_id,
                create_time=user_stock_db_record.create_time
            ) for user_stock_db_record in user_stock_db_records
        ]


def delete_user_stock(user_name: str, stock_code: str) -> bool:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return True
        else:
            user_id = user_id[0]
        user_stock_db_record: UserFavoriteStockTable | None = session.query(UserFavoriteStockTable).filter(
            UserFavoriteStockTable.user_id == user_id, UserFavoriteStockTable.stock_id == stock_code).first()
        if user_stock_db_record:
            session.delete(user_stock_db_record)
            session.commit()

    return True


def add_user_stock(user_name: str, stock_code: str) -> bool:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return True
        else:
            user_id = user_id[0]
        user_stock_db_record: UserFavoriteStockTable | None = session.query(UserFavoriteStockTable).filter(
            UserFavoriteStockTable.user_id == user_id, UserFavoriteStockTable.stock_id == stock_code).first()
        if user_stock_db_record:
            return False
        else:
            user_stock_db_record = UserFavoriteStockTable(
                stock_id=stock_code,
                user_id=user_id,
            )
            session.add(user_stock_db_record)
            session.commit()

            return True


def clear_user_stock(user_name: str) -> bool:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return True
        else:
            user_id = user_id[0]
        session.query(UserFavoriteStockTable).filter(UserFavoriteStockTable.user_id == user_id).delete()
        session.commit()
        return True

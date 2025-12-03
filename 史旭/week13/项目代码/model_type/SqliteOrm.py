# Sqlite 映射关系类（用户信息，用户收藏股票信息，聊天窗口，聊天窗口历史记录）
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column

Base = declarative_base()


# 用户信息表
class UserTable(Base):
    __tablename__ = 'user'

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    user_name: Mapped[str] = mapped_column(String(50), nullable=False)
    user_password: Mapped[str] = mapped_column(String(50), nullable=False)
    user_role: Mapped[str] = mapped_column(String(50), nullable=False)
    user_status: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(), onupdate=datetime.now())


# 用户收藏股票表
class UserStockTable(Base):
    __tablename__ = 'user_stock'

    user_stock_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.user_id"), nullable=False)
    stock_code: Mapped[str] = mapped_column(String(50), nullable=False)
    stock_name: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(), onupdate=datetime.now())


# 聊天窗口表（只保存聊天窗口的第一条信息，不记录全部聊天信息）
class ChatSessionTable(Base):
    __tablename__ = 'chat_session'

    chat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chat_title: Mapped[str] = mapped_column(Text, nullable=False)
    chat_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    update_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(), onupdate=datetime.now())


# 聊天窗口 历史记录表（保存聊天窗口 所有聊天历史记录信息）
class ChatMessageTable(Base):
    __tablename__ = 'chat_message'

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_session.chat_id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer)
    session_id: Mapped[str] = mapped_column(String(50), nullable=False)

    message_role: Mapped[str] = mapped_column(String(50), nullable=False)
    message_content: Mapped[str] = mapped_column(Text, nullable=False)
    message_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    update_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(), onupdate=datetime.now())


# 建立连接，创建表
engine = create_engine("sqlite:///./data/stock_server.db", echo=True, connect_args={"check_same_thread": False})
# Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

import datetime

from sqlalchemy.orm import sessionmaker  # 创建 数据库连接会话对象（执行sql语句，以及提交事务，关流）
from sqlalchemy import create_engine  # 创建Engine对象（用于连接数据库，SQLite、Mysql）
from sqlalchemy.ext.declarative import declarative_base  # 创建 数据库表映射的实体类
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey  # 字段类型对象
from sqlalchemy.orm import relationship  # 表与表之间的关系（一对一，一对多等）

# ORM：Object Relational Mapping  通过关联 数据库表 和 实例对象，轻松快捷实现对数据库的操作（crud）


# 1.创建连接数据库对象 Engine
# echo：将执行的sql语句 输出到控制台（便与调试）
engine = create_engine("sqlite:///C:\\ai\\sqlite\\database\\localhost.db", echo=True)

# 2. 建立连接会话 Session
# bind：绑定 engine 连接对象
Session = sessionmaker(bind=engine)
# 创建 会话 对象
session = Session()

# 3.定义实体类（映射数据库中的表）
# Base：所有映射类的父类（都需要继承，才能被 SQLalchemy 管理）
Base = declarative_base()


class KnowledgeDatabase(Base):
    # 映射的 表名
    __tablename__ = 'knowledge_database'

    # 定义字段
    # knowledge_id：Integer类型 主键 自增
    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    # 知识库标题，字符串类型 长度最大50
    knowledge_title = Column(String(50))
    # 知识库类型，字符串类型 长度最大50
    knowledge_category = Column(String(50))
    # 知识库创建时间，DateTime类型  default：新增数据时自动填充
    create_dt = Column(DateTime, default=datetime.datetime.utcnow())
    # 知识库更新时间，DateTime类型  default：新增数据时自动填充  onupdate：数据修改时自动填充
    update_dt = Column(DateTime, default=datetime.datetime.utcnow(), onupdate=datetime.datetime.utcnow())

    # knowledge_database 和 knowledge_document 之间的关系（之间获取 documents 属性，可以查询该知识库对应的文档信息）
    # 通过 主键 和 外键，区分哪个是主表，哪个是从表（即可区分 一对多等关系）
    documents = relationship("KnowledgeDocument", back_populates="knowledge")

    # 对象输出形式
    def __str__(self):
        return (f"KnowledgeDatabase(knowledge_id={self.knowledge_id}, "
                f"knowledge_title={self.knowledge_title}), "
                f"knowledge_category={self.knowledge_category}, "
                f"create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")


class KnowledegDocument(Base):
    # 映射的 表名
    __tablename__ = 'knowledge_document'

    # 定义字段
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    document_title = Column(String(50))
    document_category = Column(String(50))
    # 文档所在知识库的id（外键 关联）
    knowledge_id = Column(Integer, ForeignKey("knowledge_database.knowledge_id"))
    file_path = Column(String(50))
    file_type = Column(String(50))
    create_dt = Column(DateTime, default=datetime.datetime.utcnow())
    update_dt = Column(DateTime, default=datetime.datetime.utcnow(), onupdate=datetime.datetime.utcnow())

    # relationship 关联对象
    knowledge = relationship("KnowledgeDatabase", back_populates="documents")

    # 对象输出形式
    def __str__(self):
        return (f"KnowledegDocument(document_id={self.document_id}, "
                f"document_title={self.document_title}), "
                f"document_category={self.document_category}, "
                f"knowledge_id={self.knowledge_id}, "
                f"file_path={self.file_path}, "
                f"file_type={self.file_type}, "
                f"create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")


# Base.metadata.create_all()  创建所有class对应的表（不存在时才会创建）
# engine：连接对象，指定数据库
Base.metadata.create_all(engine)

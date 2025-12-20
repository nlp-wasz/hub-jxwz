import datetime
import time

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey,func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# 政企项目 数据库管理类（连接数据库，数据库表映射实体类，会话session）

# 1.创建 Engine 连接对象
db_url = "sqlite:///C:\\ai\\sqlite\\database\\localhost.db"
engine = create_engine(db_url, echo=True)

# 2.创建 数据库表映射实体类对象
Base = declarative_base()


class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'

    # 定义字段
    # knowledge_id：Integer类型 主键 自增
    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    # 知识库标题，字符串类型 长度最大50
    knowledge_title = Column(String(50))
    # 知识库类型，字符串类型 长度最大50
    knowledge_category = Column(String(50))
    # 知识库创建时间，DateTime类型  default：新增数据时自动填充
    create_dt = Column(DateTime, default=datetime.datetime.now)
    # 知识库更新时间，DateTime类型  default：新增数据时自动填充  onupdate：数据修改时自动填充
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

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


class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'

    # 定义字段
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    document_title = Column(String(50))
    document_category = Column(String(50))
    # 外键
    knowledge_id = Column(Integer, ForeignKey("knowledge_database.knowledge_id"))
    file_path = Column(String(50))
    file_type = Column(String(50))
    create_dt = Column(DateTime, default=datetime.datetime.now)
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

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


Base.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
# session = Session()

# session.query(KnowledgeDatabase).delete()
# session.query(KnowledgeDocument).delete()
# session.commit()

from sqlalchemy import create_engine,Column,Integer,String,ForeignKey,DateTime
from sqlalchemy.orm import sessionmaker,declarative_base,relationship
from datetime import datetime


engine=create_engine('sqlite:///library_orm.db',echo=True)

#1.创建基类
Base=declarative_base()

# 2. 定义模型类，继承自 Base
class Author(Base):
    __tablename__ = 'authors'  # 映射到数据库中的表名

    author_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    nationality = Column(String)

    # 定义与 Book 表的关系，'books' 是 Author 实例可以访问的属性
    books = relationship("Book", back_populates="author")

    def __repr__(self):
        return f"<Author(name='{self.name}', nationality='{self.nationality}')>"

class Book(Base):
    __tablename__ = 'books'

    book_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    published_year = Column(Integer)

    # 定义外键，关联到 authors 表的 author_id
    author_id = Column(Integer, ForeignKey('authors.author_id'))

    # 定义与 Author 表的关系，'author' 是 Book 实例可以访问的属性
    author = relationship("Author", back_populates="books")

    def __repr__(self):
        return f"<Book(title='{self.title}', published_year={self.published_year})>"

class Borrower(Base):
    __tablename__ = 'borrowers'

    borrower_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True)

    def __repr__(self):
        return f"<Borrower(name='{self.name}', email='{self.email}')>"


Base.metadata.create_all(engine)

Session=sessionmaker(bind=engine)
session=Session()

print("\n--- 插入数据 ---")
# 实例化模型对象
jk_rowling = Author(name='J.K. Rowling', nationality='British')
george_orwell = Author(name='George Orwell', nationality='British')
isaac_asimov = Author(name='Isaac Asimov', nationality='American')

# 将对象添加到会话中
session.add_all([jk_rowling, george_orwell, isaac_asimov])

# 插入书籍数据，通过对象关系来设置作者
book_hp = Book(title='Harry Potter', published_year=1997, author=jk_rowling)
book_1984 = Book(title='1984', published_year=1949, author=george_orwell)

session.add_all([book_hp, book_1984])

# 插入借阅人数据
borrower_alice = Borrower(name='Alice', email='alice@example.com')
borrower_bob = Borrower(name='Bob', email='bob@example.com')
session.add_all([borrower_alice, borrower_bob])

# 提交所有更改到数据库
session.commit()

results = session.query(Book).join(Author).all()
for book in results:
    print(f"书籍: {book.title}, 作者: {book.author.name}")
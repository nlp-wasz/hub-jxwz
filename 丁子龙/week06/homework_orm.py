# 导入 SQLAlchemy 所需的模块
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 创建数据库引擎，这里使用 SQLite
# check_same_thread=False 允许在多线程环境下使用，但对于单文件示例可以忽略
engine = create_engine('sqlite:///library_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()

# --- 定义 ORM 模型（与数据库表对应） ---

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

# --- 创建数据库和表 ---
# 这一步会根据上面定义的模型，在数据库中创建相应的表
Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()

# ============================
# 添加测试数据
# ============================

try:
    # 清空旧数据（可选，便于重复运行）
    print("🗑️  删除旧数据...")
    session.query(Borrower).delete()
    session.query(Book).delete()
    session.query(Author).delete()
    session.commit()
    print("旧数据已清除。\n")

    # 1. 创建作者
    print("正在创建作者...")
    author1 = Author(name="鲁迅", nationality="中国")
    author2 = Author(name="钱钟书", nationality="中国")
    author3 = Author(name="Leo Tolstoy", nationality="俄罗斯")
    author4 = Author(name="J.K. Rowling", nationality="英国")
    author5 = Author(name="F. Scott Fitzgerald", nationality="美国")

    session.add_all([author1, author2, author3, author4, author5])
    session.commit()  # 提交以确保 author_id 被分配
    print("作者创建完成。\n")

    # 2. 创建书籍（自动引用 author 对象）
    print("正在创建书籍...")
    book1 = Book(title="呐喊", published_year=1923, author=author1)
    book2 = Book(title="彷徨", published_year=1926, author=author1)
    book3 = Book(title="阿Q正传", published_year=1921, author=author1)
    book4 = Book(title="围城", published_year=1947, author=author2)
    book5 = Book(title="战争与和平", published_year=1869, author=author3)
    book6 = Book(title="哈利·波特与魔法石", published_year=1997, author=author4)
    book7 = Book(title="哈利·波特与死亡圣器", published_year=2007, author=author4)
    book8 = Book(title="了不起的盖茨比", published_year=1925, author=author5)

    session.add_all([book1, book2, book3, book4, book5, book6, book7, book8])
    session.commit()
    print("书籍创建完成。\n")

    # 3. 创建借阅者
    print("👥 正在创建借阅者...")
    borrower1 = Borrower(name="张三", email="zhangsan@lib.com")
    borrower2 = Borrower(name="李四", email="lisi@lib.com")
    borrower3 = Borrower(name="王五", email="wangwu@lib.com")
    borrower4 = Borrower(name="John Smith", email="john@lib.com")
    borrower5 = Borrower(name="Emily Chen", email="emily@lib.com")

    session.add_all([borrower1, borrower2, borrower3, borrower4, borrower5])
    session.commit()
    print("借阅者创建完成。\n")

    # ============================
    # 查询验证示例
    # ============================

    print("="*50)
    print("🔍 测试查询结果：")
    print("="*50)

    # 示例 1：列出所有作者及其作品
    print("1️⃣ 所有作者及其书籍：")
    for author in session.query(Author).all():
        print(f"📘 作者: {author.name} ({author.nationality})")
        if author.books:
            for book in author.books:
                print(f"   └──《{book.title}》({book.published_year})")
        else:
            print("   └── (暂无书籍)")
    print()

    # 示例 2：查找某本书及它的作者
    book = session.query(Book).filter_by(title="围城").first()
    if book:
        print(f"2️⃣ 《围城》信息：\n   书名: {book.title}\n   作者: {book.author.name}\n   出版年份: {book.published_year}")
    print()

    # 示例 3：统计每位作者有多少本书
    print("3️⃣ 作者图书数量统计：")
    from sqlalchemy import func
    result = session.query(
        Author.name,
        func.count(Book.book_id).label('book_count')
    ).outerjoin(Book).group_by(Author.author_id).all()

    for name, count in result:
        print(f"   {name}: {count} 本")

finally:
    # 关闭会话
    session.close()
    print("\n🔚 会话已关闭。")













































































# pip install sqlalchemy
# 导入 SQLAlchemy 所需的模块
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 创建数据库引擎，这里使用 SQLite
# check_same_thread=False 允许在多线程环境下使用，但对于单文件示例可以忽略
engine = create_engine('sqlite:///music_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()


# --- 定义 ORM 模型（与数据库表对应） ---

class Artist(Base):
    __tablename__ = 'artists'  # 映射到数据库中的表名

    artist_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    nationality = Column(String)

    # 定义与 Songs 表的关系，'songs' 是 Artist 实例可以访问的属性
    songs = relationship("Song", back_populates="artist")

    def __repr__(self):
        return f"<Artist(name='{self.name}', nationality='{self.nationality}')>"


class Song(Base):
    __tablename__ = 'songs'

    song_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    published_year = Column(Integer)

    # 定义外键，关联到 artists 表的 artist_id
    artist_id = Column(Integer, ForeignKey('artists.artist_id'))

    # 定义与 Artist 表的关系，'artist' 是 Song 实例可以访问的属性
    artist = relationship("Artist", back_populates="songs")

    def __repr__(self):
        return f"<Song(title='{self.title}', published_year={self.published_year})>"


class Listener(Base):
    __tablename__ = 'listeners'

    listener_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True)
    favorite_style = Column(String, unique=False)

    def __repr__(self):
        return f"<Listener(name='{self.name}', email='{self.email}', favorite_style='{self.favorite_style}')>"


# --- 创建数据库和表 ---
# 这一步会根据上面定义的模型，在数据库中创建相应的表
Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()

# --- 示例一：插入数据 (Create) ---
print("\n--- 插入数据 ---")
# 实例化模型对象
jielun = Artist(name='周杰伦', nationality='中国')
zongshen = Artist(name='李宗盛', nationality='中国')
isaac_asimov = Artist(name='Isaac Asimov', nationality='American')

# 将对象添加到会话中
session.add_all([jielun, zongshen, isaac_asimov])

# 插入书籍数据，通过对象关系来设置作者
song_hp = Song(title='告白气球', published_year=1997, artist=jielun)
song_1984 = Song(title='1984', published_year=1949, artist=zongshen)

session.add_all([song_hp, song_1984])

# 插入借阅人数据
listener_Tom = Listener(name='Tom', email='tom@example.com', favorite_style='pop')
listener_bob = Listener(name='Bob', email='bob@example.com', favorite_style='民谣')
session.add_all([listener_Tom, listener_bob])

# 提交所有更改到数据库
session.commit()
print("数据已成功插入。")

# --- 示例二：查询数据 (Read) ---
print("\n--- 所有书籍和它们的作者 ---")
# ORM 方式的 JOIN 查询
# 我们可以直接通过对象的属性来查询关联数据
results = session.query(Song).join(Artist).all()
for song in results:
    print(f"书籍: {song.title}, 作者: {song.artist.name}")

# --- 示例三：更新和删除数据 (Update & Delete) ---
print("\n--- 更新歌曲信息 ---")
# 查询要更新的对象
song_to_update = session.query(Song).filter_by(title='告白气球').first()
if song_to_update:
    song_to_update.published_year = 1998
    session.commit()
    print("歌曲 '告白气球' 的出版年份已更新。")

# 再次查询，验证更新
updated_song = session.query(Song).filter_by(title='告白气球').first()
if updated_song:
    print(f"更新后的信息: 歌曲: {updated_song.title}, 出版年份: {updated_song.published_year}")

print("\n--- 删除听众 ---")
# 查询要删除的对象
listener_to_delete = session.query(Listener).filter_by(name='Bob').first()
if listener_to_delete:
    session.delete(listener_to_delete)
    session.commit()
    print("听众 'Bob' 已被删除。")

# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的听众 ---")
remaining_listeners = session.query(Listener).all()
for listener in remaining_listeners:
    print(f"姓名: {listener.name}")

# 关闭会话
session.close()
print("\n会话已关闭。")

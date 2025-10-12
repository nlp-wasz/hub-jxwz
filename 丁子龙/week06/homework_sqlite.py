# sqlite
import sqlite3
conn = sqlite3.connect('test.db')
print ("数据库打开成功")
cursor = conn.cursor()
(cursor.execute
 ('''
CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT
);
'''))

# 创建 books 表，外键关联 authors 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER,
    published_year INTEGER,
    FOREIGN KEY (author_id) REFERENCES authors (author_id)
);
''')

# 创建 borrowers 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# ============================
# 插入 authors 数据
# ============================
authors_data = [
    ("鲁迅", "中国"),
    ("钱钟书", "中国"),
    ("沈从文", "中国"),
    ("Leo Tolstoy", "俄罗斯"),
    ("F. Scott Fitzgerald", "美国"),
    ("J.K. Rowling", "英国"),
    ("Gabriel García Márquez", "哥伦比亚")
]

cursor.executemany("INSERT OR IGNORE INTO authors (name, nationality) VALUES (?, ?)", authors_data)

# ============================
# 插入 books 数据（注意 author_id 必须存在）
# ============================
books_data = [
    ("呐喊", 1, 1923),
    ("彷徨", 1, 1926),
    ("阿Q正传", 1, 1921),
    ("围城", 2, 1947),
    ("边城", 3, 1934),
    ("战争与和平", 4, 1869),
    ("了不起的盖茨比", 5, 1925),
    ("哈利·波特与魔法石", 6, 1997),
    ("百年孤独", 7, 1967),
    ("哈利·波特与死亡圣器", 6, 2007)
]

cursor.executemany("INSERT OR IGNORE INTO books (title, author_id, published_year) VALUES (?, ?, ?)", books_data)

# ============================
# 插入 borrowers 数据
# ============================
borrowers_data = [
    ("张三", "zhangsan@lib.com"),
    ("李四", "lisi@lib.com"),
    ("王五", "wangwu@lib.com"),
    ("赵敏", "zhaomin@lib.com"),
    ("John Smith", "john@lib.com"),
    ("Emily Chen", "emily@lib.com")
]

cursor.executemany("INSERT OR IGNORE INTO borrowers (name, email) VALUES (?, ?)", borrowers_data)


conn.commit()

print("✅ 所有表创建成功，并已插入模拟数据！")


# 查询所有书籍及其对应的作者名字
print("\n--- 所有书籍和它们的作者 ---")
cursor.execute('''
SELECT books.title, authors.name
FROM books
JOIN authors ON books.author_id = authors.author_id;
''')

books_with_authors = cursor.fetchall()
for book, author in books_with_authors:
    print(f"书籍: {book}, 作者: {author}")

# 查询更新后的数据
cursor.execute("SELECT title, published_year FROM books WHERE title = '哈利·波特与魔法石'")
updated_book = cursor.fetchone()
print(f"更新前的信息: 书籍: {updated_book[0]}, 出版年份: {updated_book[1]}")

# 更新一本书的出版年份
print("\n--- 更新书籍信息 ---")
cursor.execute("UPDATE books SET published_year = ? WHERE title = ?",
               (1998, '哈利·波特与魔法石'))
conn.commit()
print("书籍 '哈利·波特与魔法石' 的出版年份已更新。")

# 查询更新后的数据
cursor.execute("SELECT title, published_year FROM books WHERE title = '哈利·波特与魔法石'")
updated_book = cursor.fetchone()
print(f"更新后的信息: 书籍: {updated_book[0]}, 出版年份: {updated_book[1]}")

# 再次查询借阅人列表，验证删除操作
print("\n--- 删除前的借阅人 ---")
cursor.execute("SELECT name FROM borrowers")
remaining_borrowers = cursor.fetchall()
for borrower in remaining_borrowers:
    print(f"姓名: {borrower[0]}")

# 删除一个借阅人
print("\n--- 删除借阅人 ---")
cursor.execute("DELETE FROM borrowers WHERE name = ?", ('王五',))
conn.commit()
print("借阅人 'Bob' 已被删除。")

# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的借阅人 ---")
cursor.execute("SELECT name FROM borrowers")
remaining_borrowers = cursor.fetchall()
for borrower in remaining_borrowers:
    print(f"姓名: {borrower[0]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
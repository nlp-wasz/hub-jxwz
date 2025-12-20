import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('library.db')
cursor = conn.cursor()

author_data = [
    ["鲁迅", "中国浙江"],
    ["莫言", "中国山东"],
    ["余华", "中国浙江"],
    ["贾平凹", "中国陕西"]
]

# 插入作者数据
cursor.executemany("INSERT INTO authors (name, nationality) VALUES (?, ?)", author_data)
conn.commit()

# 插入书籍数据
cursor.execute("SELECT author_id FROM authors WHERE name = '鲁迅'")
jk_rowling_id = cursor.fetchone()[0]
book_data = [
    ["狂人日记", jk_rowling_id,1918],
    ["阿Q正传", jk_rowling_id,1921],
    ["呐喊", jk_rowling_id,1923]
]
cursor.executemany("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)",
               book_data)

cursor.execute("SELECT author_id FROM authors WHERE name = '莫言'")
jk_rowling_id = cursor.fetchone()[0]

book_data = [
["红高粱家族",jk_rowling_id, 1987],
["蛙",jk_rowling_id, 2009],
["生死疲劳",jk_rowling_id, 2006]
]
cursor.executemany("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)",
               book_data)


cursor.execute("SELECT author_id FROM authors WHERE name = '余华'")
jk_rowling_id = cursor.fetchone()[0]

book_data = [
["活着", jk_rowling_id,1993],
["许三观卖血记",jk_rowling_id, 1995],
["兄弟",jk_rowling_id, 2005]
]
cursor.executemany("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)",
               book_data)

cursor.execute("SELECT author_id FROM authors WHERE name = '贾平凹'")
jk_rowling_id = cursor.fetchone()[0]

book_data = [
["废都",jk_rowling_id, 1993],
["秦腔",jk_rowling_id, 2005],
["高兴", jk_rowling_id,2007]
]
cursor.executemany("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)",
               book_data)


conn.commit()

# 插入借阅人数据
cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Alice', 'alice@example.com'))
cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Bob', 'bob@example.com'))
conn.commit()

print("数据已成功插入。")

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

# 更新一本书的出版年份
print("\n--- 更新书籍信息 ---")
cursor.execute("UPDATE books SET published_year = ? WHERE title = ?", (1998, 'Harry Potter and the Sorcerer\'s Stone'))
conn.commit()
print("书籍 'Harry Potter and the Sorcerer\'s Stone' 的出版年份已更新。")

# 查询更新后的数据
cursor.execute("SELECT title, published_year FROM books WHERE title = 'Harry Potter'")
updated_book = cursor.fetchone()
print(f"更新后的信息: 书籍: {updated_book[0]}, 出版年份: {updated_book[1]}")

# 删除一个借阅人
print("\n--- 删除借阅人 ---")
cursor.execute("DELETE FROM borrowers WHERE name = ?", ('Bob',))
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


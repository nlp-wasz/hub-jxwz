import sqlite3

conn = sqlite3.connect('library.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER,
    published_year INTEGER,
    FOREIGN KEY (author_id) REFERENCES authors (author_id)
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);
''')

conn.commit()
print("数据库和表已成功创建。")


cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('J.K. Rowling', 'British'))
cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('George Orwell', 'British'))
cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('Isaac Asimov', 'American'))
conn.commit()


cursor.execute("SELECT author_id FROM authors WHERE name = 'J.K. Rowling'")
jk_rowling_id = cursor.fetchone()[0]

cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)", ('Harry Potter', jk_rowling_id, 1997))

cursor.execute("SELECT author_id FROM authors WHERE name = 'George Orwell'")
george_orwell_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)", ('1984', george_orwell_id, 1949))

conn.commit()

cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Alice', 'alice@example.com'))
cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Bob', 'bob@example.com'))
conn.commit()

print("数据已成功插入。")

print("\n--- 所有书籍和它们的作者 ---")
cursor.execute('''
SELECT books.title, authors.name
FROM books
JOIN authors ON books.author_id = authors.author_id;
''')

books_with_authors = cursor.fetchall()
for book, author in books_with_authors:
    print(f"书籍: {book}, 作者: {author}")

print("\n--- 更新书籍信息 ---")
cursor.execute("UPDATE books SET published_year = ? WHERE title = ?", (1998, 'Harry Potter and the Sorcerer\'s Stone'))
conn.commit()
print("书籍 'Harry Potter and the Sorcerer\'s Stone' 的出版年份已更新。")

cursor.execute("SELECT title, published_year FROM books WHERE title = 'Harry Potter'")
updated_book = cursor.fetchone()
print(f"更新后的信息: 书籍: {updated_book[0]}, 出版年份: {updated_book[1]}")

print("\n--- 删除借阅人 ---")
cursor.execute("DELETE FROM borrowers WHERE name = ?", ('Bob',))
conn.commit()
print("借阅人 'Bob' 已被删除。")

print("\n--- 剩余的借阅人 ---")
cursor.execute("SELECT name FROM borrowers")
remaining_borrowers = cursor.fetchall()
for borrower in remaining_borrowers:
    print(f"姓名: {borrower[0]}")

conn.close()
print("\n数据库连接已关闭。")

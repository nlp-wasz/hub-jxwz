import sqlite3

# 1.连接对象
sqlite = sqlite3.connect("C:\\ai\\sqlite\\database\\localhost.db")

# 2.sql 执行对象
cursor = sqlite.cursor()
print("SQLite 连接成功")
print("-" * 50)

# 3.创建 test 表
cursor.execute('create table test(id INTEGER, name varchar)')
print("test表创建成功")
print("-" * 50)

# 4.添加 一条数据
insert_test = cursor.execute("insert into test values(3, '小小怪下士'),(4,'444'),(5,'555')")
print("数据添加成功")
print("-" * 50)

# 5.更新数据 id = 2  name = '快乐星球'
update_test = cursor.execute("update test set name = '快乐星球' where id = 5")
print("更新数据 id = 2  name = '快乐星球'")
print("-" * 50)

# 6.删除数据 id = 2
delete_test = cursor.execute("delete from test where id = 2")
print("删除数据 id = 2")
print("-" * 50)

# 7.查询数据（查询的数据存放在 执行sql的对象中）
cursor.execute("select * from test")
print(cursor.fetchall())

# 8.默认查询结果 是元组类型（可通过设置 连接对象 row_factory 修改返回结果形式，例如字典形式）
sqlite.row_factory = sqlite3.Row
cursor = sqlite.cursor()

cursor.execute("select * from test")
for row in cursor.fetchall():
    print(f"{row['id']}  {row['name']}")


# 9.自定义 查询结果形式（key：列名  value：列值）
def select_response(cursor, row):
    # 获取 列名
    column_size = len(cursor.description)
    column_name = []
    for i in range(column_size):
        column_name.append(cursor.description[i][0])

    # 整合 列名 和 列值
    return {col_name: row[i] for i, col_name in enumerate(column_name)}


sqlite.row_factory = select_response
cursor = sqlite.cursor()
cursor.execute("select * from test")
print(cursor.fetchall())

# 提交事务
sqlite.commit()
sqlite.close()

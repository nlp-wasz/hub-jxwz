import sqlite3

# 创建一个文件数据库 'mydatabase.db'
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# 1. 创建一个表来存储不同类型的数据
create_table_sql = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    category_name TEXT NOT NULL,             
    description TEXT,                
    created_date TEXT                     
)
"""
cursor.execute(create_table_sql)
print("数据表 'categories' 创建成功")

# 2. 插入多条不同种类的示例数据
sample_data = [
    # (category_name, description, created_date)
    ('电子产品', '手机、电脑、平板等电子设备', '2025-09-17'),
    ('服装', '上衣、裤子、鞋子等服装产品', '2025-09-17'),
    ('食品', '零食、饮料、生鲜食品', '2025-09-17'),
    ('图书', '各类书籍、杂志', '2025-09-17')
]

insert_sql = """
INSERT INTO categories
(category_name, description, created_date)
VALUES (?, ?, ?)
"""

cursor.executemany(insert_sql, sample_data)
# 插入的数据永久保存到数据库
conn.commit()
print(f"成功插入 {len(sample_data)} 条样本数据")

# 3. 检索所有数据
print("\n1. 检索所有数据:")
cursor.execute("SELECT * FROM categories")
all_rows = cursor.fetchall()
for row in all_rows:
    print(row)

# 检索特定列
print("\n2. 检索商品名称:")
cursor.execute("SELECT category_id, category_name FROM categories WHERE category_name=='服装'")
rows = cursor.fetchall()
for row in rows:
    print(f"category_id: {row[0]}, category_name: {row[1]}")

# 4. 更新数据
update_sql = "UPDATE categories SET description = '零食_01、饮料_02、生鲜食品_03'  WHERE category_name = '食品'"
cursor.execute(update_sql)
conn.commit()  # 保存更新
print("已更新 'category_name' 的分类")

# 验证更新
cursor.execute("SELECT category_name, description FROM categories WHERE category_name = '食品'")
updated_sample = cursor.fetchone()
print(f"更新后的数据: {updated_sample}")


# 5. 关闭连接
conn.close()
print("\n数据库连接已关闭。")
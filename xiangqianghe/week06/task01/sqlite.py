import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('music.db')
cursor = conn.cursor()

# 创建 artists 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS artists (
    artist_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT,
    gender TEXT NOT NULL
);
''')

# 创建 songs 表，外键关联 artists  表
cursor.execute('''
CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    artist_id INTEGER,
    release_year INTEGER,
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id)
);
''')

# 创建 listeners 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS listeners (
    listener_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    favorite_style VARCHAR
);
''')

# 提交更改
conn.commit()
print("音乐数据库和表已成功创建。")


# 插入音乐家数据
cursor.execute("INSERT INTO artists (name, nationality, gender) VALUES (?, ?, ?)", ('Taylor Swift', 'American', 'female'))
cursor.execute("INSERT INTO artists (name, nationality, gender) VALUES (?, ?, ?)", ('周杰伦', '中国', 'male'))
cursor.execute("INSERT INTO artists (name, nationality, gender) VALUES (?, ?, ?)", ('MIC', 'American', 'male'))
cursor.execute("INSERT INTO artists (name, nationality, gender) VALUES (?, ?, ?)", ('李宗盛', '中国', 'male'))
conn.commit()

# 插入歌曲数据
# 周杰伦 的 artist_id 可能是 1，我们用 SELECT 查询来获取
cursor.execute("SELECT artist_id FROM artists WHERE name = '周杰伦'")
zhou_id = cursor.fetchone()[0]

cursor.execute("INSERT INTO songs (title, artist_id, release_year) VALUES (?, ?, ?)", ('告白气球', zhou_id, 2005))

# 插入 李宗盛 的书籍
cursor.execute("SELECT artist_id FROM artists WHERE name = '李宗盛'")
zongshen_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO songs (title, artist_id, release_year) VALUES (?, ?, ?)", ('山丘', zongshen_id, 2036))

conn.commit()

# 插入听众数据
cursor.execute("INSERT OR IGNORE INTO listeners (name, email, favorite_style) VALUES (?, ?, ?)", ('Tom', 'tom@example.com', 'pop'))
cursor.execute("INSERT OR IGNORE INTO listeners (name, email, favorite_style) VALUES (?, ?, ?)", ('Bob', 'bob@example.com', '民谣'))
conn.commit()

print("数据已成功插入。")

# 查询所有歌曲及其对应的作者名字
print("\n--- 所有歌曲和它们的作者 ---")
cursor.execute('''
SELECT songs.title, artists.name
FROM songs
JOIN artists ON songs.artist_id = artists.artist_id;
''')

songs_with_artists = cursor.fetchall()
print(f'所有信息：{songs_with_artists}')
for song, artist in songs_with_artists:
    print(f"歌曲: {song}, 作者: {artist}")

# 更新歌曲的出版年份
print("\n--- 更新歌曲信息 ---")
cursor.execute("UPDATE songs SET release_year = ? WHERE title = ?", (2000, '山丘'))
conn.commit()
print("歌曲 '山丘' 的出版年份已更新。")

# 查询更新后的数据
cursor.execute("SELECT title, release_year FROM songs WHERE title = '山丘'")
updated_song = cursor.fetchone()
print(f"更新后的信息: 歌曲: {updated_song[0]}, 出版年份: {updated_song[1]}")

# 删除一个借阅人
print("\n--- 删除听众 ---")
cursor.execute("DELETE FROM listeners WHERE name = ?", ('Bob',))
conn.commit()
print("听众 'Bob' 已被删除。")

# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的听众 ---")
cursor.execute("SELECT name FROM listeners")
remaining_listeners = cursor.fetchall()
for listener in remaining_listeners:
    print(f"姓名: {listener[0]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
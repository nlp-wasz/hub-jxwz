
import redis
import json
import time
from datetime import datetime

def connect_redis():
    """连接Redis服务器"""
    try:
        # 创建Redis连接
        r = redis.Redis(
            host='localhost',
            port=6379,
            db=0,  # 使用数据库0
            decode_responses=True  # 自动解码响应
        )

        # 测试连接
        r.ping()
        print("✅ Redis连接成功!")
        print(f"Redis服务器信息: {r.info()['redis_version']}")
        return r

    except redis.ConnectionError:
        print("❌ Redis连接失败! 请确保Redis服务器正在运行")
        print("启动Redis的方法:")
        print("- Windows: 在Redis安装目录运行 redis-server.exe")
        print("- Mac: brew services start redis")
        print("- Linux: sudo systemctl start redis")
        return None

def string_operations(r):
    """String类型操作练习"""
    print("\n=== String类型操作练习 ===")

    # 设置和获取字符串
    r.set("name", "张三")
    r.set("age", "25")
    r.set("city", "北京")

    print(f"姓名: {r.get('name')}")
    print(f"年龄: {r.get('age')}")
    print(f"城市: {r.get('city')}")

    # 批量设置和获取
    r.mset({"country": "中国", "language": "中文"})
    print(f"批量获取: {r.mget('city', 'country', 'language')}")

    # 数字操作
    r.set("counter", 0)
    r.incr("counter")  # 递增1
    r.incr("counter", 5)  # 递增5
    print(f"计数器: {r.get('counter')}")

    r.decr("counter", 2)  # 递减2
    print(f"计数器递减后: {r.get('counter')}")

    # 字符串操作
    r.set("greeting", "Hello")
    print(f"字符串长度: {r.strlen('greeting')}")
    print(f"追加字符串: {r.append('greeting', ' World!')}")
    print(f"完整字符串: {r.get('greeting')}")

    # 设置过期时间（秒）
    r.setex("temp_data", 10, "这个数据10秒后过期")
    print(f"临时数据: {r.get('temp_data')}")

def list_operations(r):
    """List类型操作练习"""
    print("\n=== List类型操作练习 ===")

    # 清空之前的数据
    r.delete("mylist", "task_queue", "user_actions")

    # 1. 基本的List操作
    # 从左端添加元素（头部插入）
    r.lpush("mylist", "first", "second", "third")
    print(f"从左端添加后: {r.lrange('mylist', 0, -1)}")

    # 从右端添加元素（尾部插入）
    r.rpush("mylist", "fourth", "fifth")
    print(f"从右端添加后: {r.lrange('mylist', 0, -1)}")

    # 2. 获取列表长度
    print(f"列表长度: {r.llen('mylist')}")

    # 3. 获取指定范围的元素
    print(f"获取所有元素: {r.lrange('mylist', 0, -1)}")
    print(f"获取前3个元素: {r.lrange('mylist', 0, 2)}")
    print(f"获取后2个元素: {r.lrange('mylist', -2, -1)}")

    # 4. 从两端删除元素
    # 从左端弹出（删除并返回）
    popped_element = r.lpop("mylist")
    print(f"从左端弹出的元素: {popped_element}")
    print(f"弹出后的列表: {r.lrange('mylist', 0, -1)}")

    # 从右端弹出
    popped_element = r.rpop("mylist")
    print(f"从右端弹出的元素: {popped_element}")
    print(f"弹出后的列表: {r.lrange('mylist', 0, -1)}")

    # 高级操作
    print("\n=== List高级操作 ===")

    # 重新设置一个列表用于演示
    r.delete("students")
    r.rpush("students", "张三", "李四", "王五", "赵六", "钱七")
    print(f"学生列表: {r.lrange('students', 0, -1)}")

    # 通过索引获取元素
    print(f"第1个学生(索引0): {r.lindex('students', 0)}")
    print(f"第3个学生(索引2): {r.lindex('students', 2)}")
    print(f"最后1个学生(索引-1): {r.lindex('students', -1)}")

    # 设置指定索引的元素
    r.lset("students", 2, "王小五")
    print(f"修改后: {r.lrange('students', 0, -1)}")

    # 消息队列模拟
    print("\n=== 消息队列模拟 ===")
    r.delete("task_queue")
    tasks = ["用户注册", "发送邮件", "数据备份", "生成报表"]
    for task in tasks:
        r.rpush("task_queue", task)

    print(f"任务队列: {r.lrange('task_queue', 0, -1)}")

    # 模拟消费者处理任务
    task = r.lpop("task_queue")
    while task:
        print(f"处理任务: {task}")
        time.sleep(0.1)  # 模拟任务处理
        task = r.lpop("task_queue")

    print(f"所有任务处理完成，队列剩余: {r.llen('task_queue')}")

    # 列表修剪
    r.delete("logs")
    for i in range(10):
        r.rpush("logs", f"日志{i+1}")

    print(f"原始日志: {r.lrange('logs', 0, -1)}")
    r.ltrim("logs", 1, 6)  # 只保留索引1到6的元素
    print(f"修剪后日志: {r.lrange('logs', 0, -1)}")

    # 移除指定值的元素
    r.delete("numbers")
    r.rpush("numbers", "1", "2", "3", "2", "4", "2", "5")
    print(f"数字列表: {r.lrange('numbers', 0, -1)}")

    removed_count = r.lrem("numbers", 0, "2")  # 移除所有"2"
    print(f"移除了{removed_count}个'2'")
    print(f"移除后: {r.lrange('numbers', 0, -1)}")

def set_operations(r):
    """Set类型操作练习"""
    print("\n=== Set类型操作练习 ===")

    # 清空之前的数据
    r.delete("myset", "user1_tags", "user2_tags", "all_users", "admin_users")

    # 1. 基本的Set操作
    r.sadd("myset", "apple", "banana", "orange")
    print(f"添加元素后: {r.smembers('myset')}")

    # 重复添加元素（会被忽略）
    r.sadd("myset", "apple", "grape", "apple")
    print(f"重复添加后: {r.smembers('myset')}")

    # 2. 集合成员操作
    print(f"集合大小: {r.scard('myset')}")
    print(f"'apple'是否存在: {r.sismember('myset', 'apple')}")
    print(f"'watermelon'是否存在: {r.sismember('myset', 'watermelon')}")

    # 3. 随机操作
    print(f"随机获取一个元素: {r.srandmember('myset')}")
    print(f"随机弹出元素: {r.spop('myset')}")
    print(f"弹出后集合: {r.smembers('myset')}")

    # 4. 移除指定元素
    r.srem("myset", "banana")
    print(f"移除'banana'后: {r.smembers('myset')}")

    # 集合运算和应用场景
    print("\n=== Set集合运算 ===")

    # 用户标签系统
    r.delete("user1_tags")
    r.sadd("user1_tags", "编程", "Python", "音乐", "摄影", "旅游")
    print(f"用户1的标签: {r.smembers('user1_tags')}")

    r.delete("user2_tags")
    r.sadd("user2_tags", "Python", "JavaScript", "音乐", "绘画", "旅游")
    print(f"用户2的标签: {r.smembers('user2_tags')}")

    # 集合运算
    print(f"\n共同的标签(交集): {r.sinter('user1_tags', 'user2_tags')}")
    print(f"所有标签(并集): {r.sunion('user1_tags', 'user2_tags')}")
    print(f"用户1独有标签(差集): {r.sdiff('user1_tags', 'user2_tags')}")
    print(f"用户2独有标签(差集): {r.sdiff('user2_tags', 'user1_tags')}")

    # 将运算结果存储到新集合
    r.sinterstore("common_tags", "user1_tags", "user2_tags")
    print(f"存储的共同标签: {r.smembers('common_tags')}")

    # 权限控制应用
    print("\n=== 权限控制示例 ===")

    r.delete("all_users")
    r.sadd("all_users", "alice", "bob", "charlie", "david", "eve")
    print(f"所有用户: {r.smembers('all_users')}")

    r.delete("admin_users")
    r.sadd("admin_users", "alice", "charlie")
    print(f"管理员用户: {r.smembers('admin_users')}")

    # 检查用户权限
    test_users = ["alice", "bob", "charlie", "frank"]
    for user in test_users:
        is_admin = r.sismember("all_users", user) and r.sismember("admin_users", user)
        status = '管理员' if is_admin else '普通用户' if r.sismember("all_users", user) else '不存在'
        print(f"用户 {user}: {status}")

    # 去重应用
    print("\n=== 去重应用示例 ===")
    r.delete("unique_visitors")
    visitors = ["user1", "user2", "user1", "user3", "user2", "user4", "user1"]
    for visitor in visitors:
        r.sadd("unique_visitors", visitor)

    print(f"原始访问记录: {visitors}")
    print(f"独立访客: {r.smembers('unique_visitors')}")
    print(f"总访问次数: {len(visitors)}")
    print(f"独立访客数: {r.scard('unique_visitors')}")

def hash_operations(r):
    """Hash类型操作练习"""
    print("\n=== Hash类型操作练习 ===")

    # 清空之前的数据
    r.delete("user:1001", "product:001", "config:app")

    # 1. 基本的Hash操作
    r.hset("user:1001", "name", "张三")
    r.hset("user:1001", "age", "25")
    r.hset("user:1001", "email", "zhangsan@example.com")
    r.hset("user:1001", "city", "北京")

    print(f"用户所有信息: {r.hgetall('user:1001')}")

    # 获取单个字段值
    print(f"用户姓名: {r.hget('user:1001', 'name')}")
    print(f"用户年龄: {r.hget('user:1001', 'age')}")

    # 检查字段是否存在
    print(f"字段'name'是否存在: {r.hexists('user:1001', 'name')}")
    print(f"字段'phone'是否存在: {r.hexists('user:1001', 'phone')}")

    # 2. 批量操作
    user_info = {
        "id": "1001",
        "username": "zhangsan",
        "role": "user",
        "status": "active"
    }
    r.hmset("user:1001", user_info)
    print(f"批量设置后: {r.hgetall('user:1001')}")

    # 批量获取多个字段
    fields = ["name", "age", "email"]
    values = r.hmget("user:1001", *fields)
    print(f"批量获取字段: {dict(zip(fields, values))}")

    # 3. 字段操作
    print(f"所有字段名: {r.hkeys('user:1001')}")
    print(f"所有字段值: {r.hvals('user:1001')}")
    print(f"字段数量: {r.hlen('user:1001')}")

    # 4. 字段的增删改
    # 修改字段值
    r.hset("user:1001", "age", "26")
    print(f"修改年龄后: {r.hget('user:1001', 'age')}")

    # 增加新字段
    r.hset("user:1001", "phone", "13800138000")
    print(f"添加电话后: {r.hgetall('user:1001')}")

    # 删除字段
    r.hdel("user:1001", "city")
    print(f"删除城市字段后: {r.hgetall('user:1001')}")

    # 数字字段操作
    r.hset("user:1001", "login_count", "0")
    r.hincrby("user:1001", "login_count", 1)  # 递增1
    print(f"登录次数递增后: {r.hget('user:1001', 'login_count')}")

    # 应用场景：商品信息管理
    print("\n=== 商品信息管理示例 ===")
    r.delete("product:001")
    product_info = {
        "id": "001",
        "name": "iPhone 15",
        "price": "5999",
        "category": "手机",
        "stock": "100",
        "description": "最新款苹果手机"
    }
    r.hmset("product:001", product_info)

    print(f"商品信息: {r.hgetall('product:001')}")

    # 更新库存
    current_stock = int(r.hget("product:001", "stock"))
    new_stock = current_stock - 5  # 卖出5台
    r.hset("product:001", "stock", str(new_stock))
    print(f"更新后库存: {r.hget('product:001', 'stock')}")

def sorted_set_operations(r):
    """Sorted Set类型操作练习"""
    print("\n=== Sorted Set类型操作练习 ===")

    # 清空之前的数据
    r.delete("leaderboard", "user_scores", "timeline")

    # 1. 基本的Sorted Set操作
    # 添加成员和分数
    r.zadd("leaderboard", {"张三": 95, "李四": 87, "王五": 92, "赵六": 78})
    print(f"排行榜所有成员: {r.zrange('leaderboard', 0, -1, withscores=True)}")

    # 获取成员分数
    print(f"张三的分数: {r.zscore('leaderboard', '张三')}")

    # 获取排名
    print(f"张三的排名: {r.zrevrank('leaderboard', '张三')}")  # 从高到低的排名

    # 2. 排序操作
    print(f"按分数从低到高: {r.zrange('leaderboard', 0, -1, withscores=True)}")
    print(f"按分数从高到低: {r.zrevrange('leaderboard', 0, -1, withscores=True)}")

    # 3. 范围操作
    # 获取指定分数范围的成员
    print(f"分数80-90的成员: {r.zrangebyscore('leaderboard', 80, 90, withscores=True)}")

    # 获取指定排名范围的成员
    print(f"前2名: {r.zrevrange('leaderboard', 0, 1, withscores=True)}")

    # 4. 分数操作
    # 增加成员分数
    r.zincrby("leaderboard", 5, "李四")  # 李四加5分
    print(f"李四加分后: {r.zscore('leaderboard', '李四')}")

    # 5. 应用场景：时间线
    print("\n=== 时间线应用示例 ===")

    # 使用时间戳作为分数
    current_time = int(time.time())
    posts = [
        (current_time - 3600, "用户A发布了新文章"),
        (current_time - 1800, "用户B分享了一张图片"),
        (current_time - 900, "用户C更新了状态"),
        (current_time, "用户D发布了视频")
    ]

    r.delete("timeline")
    for timestamp, content in posts:
        r.zadd("timeline", {content: timestamp})

    # 按时间倒序获取最新动态
    print(f"最新动态: {r.zrevrange('timeline', 0, -1)}")

    # 获取最近1小时的动态
    one_hour_ago = current_time - 3600
    recent_posts = r.zrangebyscore("timeline", one_hour_ago, current_time)
    print(f"最近1小时动态: {recent_posts}")
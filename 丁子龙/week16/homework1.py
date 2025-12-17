import redis
import json

# 连接到Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)


# String类型
def demo_string():
    print("=== String类型 ===")

    # 设置键值
    r.set('name', '张三')
    r.set('age', '25')
    r.setex('temp_key', 60, '临时数据')  # 60秒后过期

    # 获取值
    name = r.get('name')
    print(f"姓名: {name}")

    # 自增/自减
    r.set('counter', 0)
    r.incr('counter')  # +1
    r.incrby('counter', 5)  # +5
    r.decr('counter')  # -1

    counter = r.get('counter')
    print(f"计数器: {counter}")

    # 批量操作
    r.mset({'city': '北京', 'country': '中国'})
    values = r.mget('city', 'country')
    print(f"批量获取: {values}")


# List类型
def demo_list():
    print("\n=== List类型 ===")

    # 添加元素
    r.delete('mylist')  # 先清除
    r.lpush('mylist', 'first')  # 左侧插入
    r.rpush('mylist', 'second')  # 右侧插入
    r.lpush('mylist', 'zero')

    # 获取列表
    all_items = r.lrange('mylist', 0, -1)
    print(f"全部元素: {all_items}")

    # 弹出元素
    left_item = r.lpop('mylist')
    right_item = r.rpop('mylist')
    print(f"左侧弹出: {left_item}, 右侧弹出: {right_item}")

    # 列表长度
    length = r.llen('mylist')
    print(f"列表长度: {length}")

# Set类型
def demo_set():
    print("\n=== Set类型 ===")

    # 添加元素
    r.sadd('tags', 'python', 'redis', 'database', 'cache')
    r.sadd('tags2', 'python', 'java', 'javascript')

    # 获取所有元素
    tags = r.smembers('tags')
    print(f"所有标签: {tags}")

    # 集合运算
    intersection = r.sinter('tags', 'tags2')  # 交集
    union = r.sunion('tags', 'tags2')  # 并集
    diff = r.sdiff('tags', 'tags2')  # 差集

    print(f"交集: {intersection}")
    print(f"并集: {union}")
    print(f"差集: {diff}")

    # 随机弹出
    random_tag = r.spop('tags')
    print(f"随机弹出: {random_tag}")

# ZSet有序集合
def demo_zset():
    print("\n=== ZSet有序集合 ===")

    # 添加带分数的成员
    r.zadd('rank', {
        'Alice': 95,
        'Bob': 87,
        'Charlie': 92,
        'David': 78
    })

    # 按分数排序获取
    top3 = r.zrevrange('rank', 0, 2, withscores=True)
    print(f"前三名: {top3}")

    # 获取分数范围
    good_students = r.zrangebyscore('rank', 90, 100, withscores=True)
    print(f"90分以上: {good_students}")

    # 获取排名
    bob_rank = r.zrevrank('rank', 'Bob')
    bob_score = r.zscore('rank', 'Bob')
    print(f"Bob排名: {bob_rank}, 分数: {bob_score}")

if __name__ == "__main__":
    # 测试连接
    try:
        response = r.ping()
        print(f"Redis连接成功: {response}")
        demo_string()
        demo_list()
        demo_set()
        demo_zset()
    except redis.exceptions.ConnectionError as e:
        print(f"连接失败: {e}")

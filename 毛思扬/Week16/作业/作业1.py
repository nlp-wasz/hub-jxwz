import redis
import time

# 连接池（推荐生产环境使用）
pool = redis.ConnectionPool(
    host='127.0.0.1',
    port=6379,
    max_connections=10,
    decode_responses=True,
    db=1,
    password='123456'
)


def advanced_features():
    r = redis.Redis(connection_pool=pool)

    print("\n1. List操作:")

    # 1. 向列表左侧插入元素 (LPUSH)
    r.lpush('mylist', 'item1', 'item2', 'item3')
    # 结果: ['item3', 'item2', 'item1']

    # 2. 向列表右侧插入元素 (RPUSH)
    r.rpush('mylist', 'item4')
    # 结果: ['item3', 'item2', 'item1', 'item4']

    # 3. 从列表左侧弹出元素 (LPOP)
    left_item = r.lpop('mylist')
    print(f"从左侧弹出: {left_item}")

    # 4. 从列表右侧弹出元素 (RPOP)
    right_item = r.rpop('mylist')
    print(f"从右侧弹出: {right_item}")

    # 5. 获取列表长度 (LLEN)
    length = r.llen('mylist')
    print(f"列表长度: {length}")

    # 6. 获取列表指定范围的元素 (LRANGE)
    items = r.lrange('mylist', 0, -1)  # 获取所有元素
    print(f"列表所有元素: {items}")

    print("\n1. Set操作:")

    # 1. 向集合添加元素 (SADD)
    r.sadd('myset', 'member1', 'member2', 'member3')
    r.sadd('myset', 'member4')

    # 2. 检查元素是否存在于集合中 (SISMEMBER)
    is_member = r.sismember('myset', 'member1')
    print(f"'member1' 是否在集合中: {is_member}")

    # 3. 获取集合所有成员 (SMEMBERS)
    members = r.smembers('myset')
    print(f"集合所有成员: {members}")

    # 4. 从集合中移除元素 (SREM)
    r.srem('myset', 'member2')

    # 5. 获取集合大小 (SCARD)
    size = r.scard('myset')
    print(f"集合大小: {size}")

    # 6. 集合间操作
    r.sadd('set1', 'a', 'b', 'c')
    r.sadd('set2', 'b', 'c', 'd')

    # 交集 (SINTER)
    intersection = r.sinter('set1', 'set2')
    print(f"交集: {intersection}")

    # 并集 (SUNION)
    union = r.sunion('set1', 'set2')
    print(f"并集: {union}")

    # 差集 (SDIFF)
    difference = r.sdiff('set1', 'set2')
    print(f"差集: {difference}")

# 运行高级功能
if __name__ == "__main__":
    advanced_features()

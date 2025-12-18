# 连接本地 Redis
import redis.exceptions
from redis import Redis

r = Redis(
    host='localhost',
    port=6379,
    db=0,
    password=None,
    decode_responses=True
)

try:
    p = r.ping()
    print(f"连接成功：{p}")
except redis.exceptions.ConnectionError as e:
    print(e)

keys = r.keys("sys_dict:*")
for key in keys:
    va = r.get(key)
    print(f"{key} -> {va}")

# Redis 中数据存储方式为 key-value 形式（可以存储在不同的分组下面）
# 而 key 为字符串，value 包含五种数据类型（字符串str，双向链表List，哈希映射Hash，无序不重复集合Set，有序带分数集合ZSet）
from redis import Redis

# 连接 Redis
r = Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 指定 key 的分组
group_str = "redis_str"
group_list = "redis_list"
group_hash = "redis_hash"
group_set = "redis_set"
group_zset = "redis_zset"


# 1.字符串类型（读写操作）
def redis_str():
    # set 添加 kv
    r.set(f"{group_str}:name", "小旭")

    # setex 添加带过期时间的 kv
    r.setex(f"{group_str}:age", 100, 10)

    # mset() 批量增加 kv
    r.mset({f"{group_str}:sex": "男", f"{group_str}:TTL": 100})

    # get 根据 key 获取 value
    name = r.get(f"{group_str}:name")
    print(f"{group_str}:name：{name}")

    # r.mget() 获取多个 key 信息
    mget = r.mget([f"{group_str}:sex", f"{group_str}:TTL"])
    print(f"mget:{mget}")

    # value是 数字类型，可进行 自增，自减
    print(f"{group_str}:age 自增-自减 前：{r.get(f"{group_str}:age")}")
    r.incr(f"{group_str}:age")  # +1
    r.incrby(f"{group_str}:age", 5)  # +5
    r.decr(f"{group_str}:age")  # -1
    r.decrby(f"{group_str}:age", 5)  # -5
    print(f"{group_str}:age 自增-自减 后：{r.get(f"{group_str}:age")}")


# 2.双向链表 List 类型（读写操作）
def redis_list():
    # push 添加 kv
    r.lpush(f"{group_list}:list", "小旭list")
    r.rpush(f"{group_list}:list", 20)
    r.lpush(f"{group_list}:list", "男")

    # pop 弹出元素
    sex = r.lpop(f"{group_list}:list")
    age = r.rpop(f"{group_list}:list")
    name = r.lpop(f"{group_list}:list")
    print(f"name:{name}, age:{age}, sex:{sex}")

    # lrange 从左侧开始获取元素（起止索引）
    all_list = r.lrange(f"{group_list}:list", 0, -1)
    print(f"all_list:{all_list}")

    # rlen 获取 list 长度
    len_list = r.llen(f"{group_list}:list")
    print(f"len_list:{len_list}")


# 3.哈希 Hash 类型（读写操作）
def redis_hash():
    # hset 添加 kv
    r.hmset(
        name=f"{group_hash}:hash1",
        mapping={
            "name": "hash",
            "age": 25,
            "sex": "女"
        }
    )

    # hget 根据 key + hash mapping 中的 key 获取 value
    name = r.hget(name=f"{group_hash}:hash1", key="name")
    print(f"name:{name}")

    # hgetall 获取 key 中的 hash mapping 映射对象
    all_hash = r.hgetall(f"{group_hash}:hash1")
    print(f"all_hash:{all_hash}")

    # hincrby 对 mapping 中的数值类型，进行自增
    print(f"{group_hash}:hash1-age 自增前：{r.hget(f"{group_hash}:hash1", "age")}")
    r.hincrby(f"{group_hash}:hash1", "age", 5)  # +5
    print(f"{group_hash}:hash-age 自增后：{r.hget(f"{group_hash}:hash1", "age")}")

    # hexists 判断 mapping 中指定字段是否存在
    if not r.hexists(f"{group_hash}:hash1", "TTL"):
        print(f"TTL不存在hash mapping中")


# 4.无序不重复集合 Set 类型（读写操作）
def redis_set():
    # hset 添加 kv
    r.sadd(f"{group_set}:set1", "小旭", 100, "男", 100)
    r.sadd(f"{group_set}:set2", "小旭", "女")

    # smembers 根据key 获取 集合信息
    all_set = r.smembers(f"{group_set}:set1")
    print(f"all_set:{all_set}")

    # 集合的 交集、并集、差集
    sinter = r.sinter([f"{group_set}:set1", f"{group_set}:set2"])
    sunion = r.sunion([f"{group_set}:set1", f"{group_set}:set2"])
    sdiff = r.sdiff([f"{group_set}:set1", f"{group_set}:set2"])
    print(f"sinter:{sinter}")
    print(f"sunion:{sunion}")
    print(f"sdiff:{sdiff}")


# 5.有序可重复集合 ZSet 类型（读写操作）
def redis_zset():
    # zset 添加 kv
    r.zadd(f"{group_zset}:set1", {
        "小旭": 5,
        100: 2,
        "男": 4,
        "山东": 9
    })

    # zrevrange 根据score 对key中的mapping进行排序，之后根据起止索引，获取元素信息
    zrevrange = r.zrevrange(f"{group_zset}:set1", 0, -1, withscores=True)
    print(f"zrevrange:{zrevrange}")

    # zrangebyscore 获取 指定分数范围内的 元素信息
    zrangebyscore = r.zrangebyscore(f"{group_zset}:set1", 5, 10, withscores=True)
    print(f"zrangebyscore:{zrangebyscore}")

    # 获取 指定元素的排名和分数
    zrank = r.zrank(f"{group_zset}:set1", "小旭")
    zscore = r.zscore(f"{group_zset}:set1", "小旭")
    print(f"小旭 排名: {zrank}, 分数: {zscore}")


if __name__ == '__main__':
    # 1.字符串类型（读写操作）
    redis_str()
    print("-" * 50)

    # 2.双向链表 List 类型（读写操作）
    redis_list()
    print("-" * 50)

    # 3.哈希 Hash 类型（读写操作）
    # redis_hash()
    print("-" * 50)

    # 4.无序不重复集合 Set 类型（读写操作）
    redis_set()
    print("-" * 50)

    # 5.有序可重复集合 ZSet 类型（读写操作）
    redis_zset()
    print("-" * 50)

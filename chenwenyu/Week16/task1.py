import redis
import json

# 连接到Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# String类型
def demo_string():
    print("=== String类型 ===")
    
    # 设置键值
    r.set('name', 'wenyuc')
    r.set('age', '25')
    r.setex('temp_key', 60, '临时数据')  # 60秒后过期
    
    # 获取值
    name = r.get('name')
    print(f"姓名: {name}")
    
    # 自增/自减
    r.set('counter', 0)
    r.incr('counter')      # +1
    r.incrby('counter', 5) # +5
    r.decr('counter')      # -1
    
    counter = r.get('counter')
    print(f"计数器: {counter}")
    
    # 批量操作
    r.mset({'city': '北京', 'country': '中国'})
    r.mset({'city': 'Paris', 'country':'France'})
    
    values = r.mget('city', 'country')
    print(f"批量获取: {values}")

# List类型
def demo_list():
    print("\n=== List类型 ===")
    
    # 添加元素
    r.delete('mylist')  # 先清除
    r.lpush('mylist', 'first')     # 左侧插入
    r.rpush('mylist', 'second')    # 右侧插入
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

# Hash类型
def demo_hash():
    print("\n=== Hash类型 推荐存储对象===")
    
    # 设置哈希字段
    user_key = 'user:1001'
    r.hset(user_key, mapping={
        'name': '李四',
        'age': 30,
        'email': 'lisi@example.com',
        'city': '上海'
    })
    
    # 获取单个字段
    name = r.hget(user_key, 'name')
    print(f"用户名: {name}")
    
    # 获取所有字段
    user_data = r.hgetall(user_key)
    print(f"用户数据: {user_data}")
    
    # 修改字段
    r.hincrby(user_key, 'age', 1)  # 年龄+1
    
    # 检查字段是否存在
    has_email = r.hexists(user_key, 'email')
    print(f"是否有email字段: {has_email}")

# Set类型
def demo_set_correct():
    """正确的Set演示代码"""
    
    # 先清理测试数据
    keys_to_delete = ['tags', 'tags2', 'temp_set', 'temp_set2']
    r.delete(*keys_to_delete)
    
    print("=== Set类型演示 ===")
    
    # 添加元素到tags集合
    added1 = r.sadd('tags', 'python', 'redis', 'database', 'cache')
    print(f"向'tags'添加元素，成功添加了 {added1} 个新元素")
    
    # 添加元素到tags2集合
    added2 = r.sadd('tags2', 'python', 'java', 'javascript')
    print(f"向'tags2'添加元素，成功添加了 {added2} 个新元素")
    
    # 获取所有元素
    tags = r.smembers('tags')
    tags2 = r.smembers('tags2')
    
    print(f"\n'tags'集合 ({len(tags)}个元素): {tags}")
    print(f"'tags2'集合 ({len(tags2)}个元素): {tags2}")
    
    # 集合运算
    print("\n=== 集合运算 ===")
    
    # 交集：两个集合都有的元素
    common = r.sinter('tags', 'tags2')
    print(f"交集（共同标签）: {common}")
    
    # 并集：所有不重复的元素
    all_tags = r.sunion('tags', 'tags2')
    print(f"并集（所有标签）: {all_tags}")
    
    # 差集：在tags中但不在tags2中的元素
    only_in_tags = r.sdiff('tags', 'tags2')
    print(f"差集（只在tags中）: {only_in_tags}")
    
    # 随机获取元素
    random_tag = r.srandmember('tags')
    print(f"\n随机获取一个标签: {random_tag}")
    
    # 检查元素是否存在
    has_python = r.sismember('tags', 'python')
    has_java = r.sismember('tags', 'java')
    print(f"'tags'包含'python'吗? {has_python}")
    print(f"'tags'包含'java'吗? {has_java}")
    
    # 移除元素
    removed = r.srem('tags', 'cache')
    print(f"\n从'tags'移除'cache'，成功移除了 {removed} 个元素")
    
    tags_after_removal = r.smembers('tags')
    print(f"移除后的'tags': {tags_after_removal}")

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
    print(f"Bob排名: {bob_rank+1}, 分数: {bob_score}")

"""ZSET中的排序
命令	默认排序	说明
ZRANGE	升序（从小到大）	按分数从低到高排序
ZREVRANGE	降序（从小到大）	按分数从高到低排序
ZRANGEBYSCORE	升序	按分数从低到高排序
ZREVRANGEBYSCORE	降序	按分数从高到低排序
ZPOPMIN	升序	弹出分数最低的元素
ZPOPMAX	降序	弹出分数最高的元素
"""

# 使用普通 Redis + JSON 字符串
class SimpleRedisJSON:
    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    def json_set(self, key, value):
        """存储 JSON 数据"""
        self.r.set(key, json.dumps(value, ensure_ascii=False))
    
    def json_get(self, key, path=None):
        """获取 JSON 数据，支持简单路径"""
        data = json.loads(self.r.get(key))
        if path:
            # 简单的路径解析，如 "user.name"
            parts = path.split('.')
            for part in parts:
                if part in data:
                    data = data[part]
                else:
                    return None
        return data

def demo_simple_json():
    # 使用示例
    store = SimpleRedisJSON()
    store.json_set('user:1', {'name': 'wenyu', 'age': 25, 'address': {'city': '北京', 'country': 'China'}})
    store.json_set('user:2', {'name': 'wangxue', 'age': 23, 'address': {'city': 'Paris', 'country': 'France'}})
    print(store.json_get('user:1', 'address.city'))  # 输出: 北京
    print(store.json_get('user:2', 'address.city'))  # 输出: Paris
        
# 运行所有示例
def run_all_demos():
    demo_string()
    demo_list()
    demo_hash()
    demo_set_correct()
    demo_zset()
    demo_simple_json()

if __name__ == "__main__":
    run_all_demos()
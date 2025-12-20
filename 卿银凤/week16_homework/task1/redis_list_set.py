import redis

# 连接到Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)


def demo_list():
    r.lpush('mylist', 'first')  # 左侧插入
    r.rpush('mylist', 'second')  # 右侧插入
    # 获取列表
    all_items = r.lrange('mylist', 0, -1)
    print(f"全部元素: {all_items}")
    # 弹出元素
    left_item = r.lpop('mylist')
    right_item = r.rpop('mylist')
    print(f"左侧弹出: {left_item}, 右侧弹出: {right_item}")
    # 列表长度
    length = r.llen('mylist')
    print(length)


def demo_set():
    # 添加元素
    r.sadd('tags', 'python', 'redis', 'database', 'cache')
    # 获取所有元素
    tags = r.smembers('tags')
    print(f"所有标签: {tags}")


def run_all_demos():
    demo_list()
    demo_set()


if __name__ == "__main__":
    run_all_demos()
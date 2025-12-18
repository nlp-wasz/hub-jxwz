import redis
# 连接到Redis服务器
r = redis.Redis(host='ip', port=6379, db=0, decode_responses=True)

r.mset({'age': 30, 'city': 'Beijing'})
r.set('name', 'lee')
r.lpush('mylist', 'a', 'b', 'c')
r.rpush('mylist', 'd')

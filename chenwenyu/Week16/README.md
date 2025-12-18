## Week16 Homework

### task1.py

- 本地安装redis server
- 实现Redis string，list，hash，set, zset, json

### task2.py

-阅读semanticCache.py,找出其中的bug
主要bug在于self.redis.delete(*pormpts)，需要删除键名，而不是提示文本字符串。

- 修改bug，并改进了其他一些代码

### task3.py
- 基于 Redis 实现语义路由，一个完整的伪代码实现。这个实现使用 Redis 存储路由规则，并支持语义匹配（基于向量相似度）
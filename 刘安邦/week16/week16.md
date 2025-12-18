## 1、本地安装一下redis （或使用 http://onecompiler.com/redis/），学习redis的使用，学会操作list、set。
### List
- **增**
  - lpush, rpush, 返回List大小
- **删**
  - 对单元素：lpop, rpop, 返回元素
  - 对List：delete, 返回1成功0失败
- **查**
  - lrange('MyList',0,-1)获取全部元素（注意索引范围）, llen获取List大小
### Set
- **增**
  - sadd('setname', values)
- **删**
  - spop('setname')随机弹出
  - 对Set：delete, 返回1成功0失败
- **查**
  - smembers('setname')获取全部元素, scard('setname')获取Set成员数
- **运算(交并差)**
  - sinter, sunion, sdiff

## 2、阅读下 https://github.com/redis/redis-vl-python 的项目和已有的代码，阅读 SemanticCache.py，找出其中存在bug。
### a 索引误用
```python
filtered_prompts = [pormpts[i] for i in filtered_ind]
```
- 把来自`enumerate(dis[0])`的顺序索引`i`，修改为来自faiss检索出的向量索引`ind`。
- faiss存储新数据在末尾，redis用lpush存储新数据在列表头，顺序相反

最终调整为
```python
n_total = len(pormpts)
filtered_prompts = [pormpts[n_total-1-ind[0][i]] for i in filtered_ind]
```

### b 删除键值错误
问答的键值对，键带有前缀`self.name + "key:`，而在clear_cache函数中获取到的`pormpts`提问不带前缀。
因此应把如下删除键值对操作加以修改。
```python
# 修改前
self.redis.delete(*pormpts)
# 修改后
for p in pormpts:
    key = self.name + "key:" + p.decode()
    self.redis.delete(key)
```

## 3、基于已有的代码，完成SemanticRouter （快速实现意图识别）的逻辑和框架，可以写伪代码（运行不起来也没问题也可以）。
见SemanticRouter.py

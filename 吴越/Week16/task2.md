## 问题1
FAISS存储的问题索引和Redis列表存储的问题索引对应关系是相反的.
FAISS向量数据库中通过这种方式添加问题"self.index.add(embedding)"最新的问题在最后，
Redis数据库 中通过pipe.lpush(self.name + "list", q)方式添加问题，最新的问题在最前面，最终会导致FAISS和Redis数据库中的问题不匹配
导致无法找到正确的response

## 问题2
self.redis.delete(*pormpts)，这里删除的是prompt内容作为键，而不是实际存储response的键！删除了错误的键，response数据还留在Redis中！

```commandline

def clear_cache(self):
    #从Redis列表中获取所有存储的prompt
    pormpts = self.redis.lrange(self.name + "list", 0, -1)
    #这里删除的是prompt内容作为键，而不是实际存储response的键！删除了错误的键，response数据还留在Redis中！
    self.redis.delete(*pormpts)
    #删除存储所有prompt的列表
    self.redis.delete(self.name + "list")
    os.unlink(f"{self.name}.index")
    self.index = None


正确处理方式：
def clear_cache(self):
    #从Redis列表中获取所有存储的prompt
    pormpts = self.redis.lrange(self.name + "list", 0, -1)
    Key_to_delete=[]
    for prompt in prompts:
        prompt=prompt.decode()
        key=f"{self.name}+"key:"+prompt"
        Key_to_delete.append(key)
    
    self.redis.delete(*Key_to_delete)
    #删除存储所有prompt的列表
    self.redis.delete(self.name + "list")
    os.unlink(f"{self.name}.index")
    self.index = None

```
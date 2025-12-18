## 主要Bug

### 1. **索引与Redis数据不一致问题**

```
# SemanticCache.py代码
filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
pormpts = self.redis.lrange(self.name + "list", 0, -1)
filtered_prompts = [pormpts[i] for i in filtered_ind]
```

**问题**：FAISS索引的顺序与Redis列表的顺序可能不一致。FAISS索引是按添加顺序分配的，但Redis列表使用`lpush`，新数据在列表头部。

### 2. **距离阈值逻辑错误**

```
if dis[0][0] > self.distance_threshold:
    return None
```

**问题**：这里检查的是第一个结果的距离，但后续又过滤所有小于阈值的结果。逻辑矛盾。

### 3. **键名冲突风险**

```
pipe.setex(self.name + "key:" + q, self.ttl, a)
```

**问题**：如果prompt内容相同但属于不同会话，会发生键冲突。

### 4. **内存泄漏风险**

Redis列表`self.name + "list"`会无限增长，没有清理机制。
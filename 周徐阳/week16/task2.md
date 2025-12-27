## 主要 Bug 列表

### 1. **严重 Bug：索引顺序不一致**
```python
# store 方法中
pipe.lpush(self.name + "list", q)  # ❌ lpush 从头部插入，顺序会反转
```

**问题**：
- `lpush` 从列表头部插入，后添加的元素会在前面
- FAISS 索引是按添加顺序递增的（0, 1, 2, ...）
- 这导致 FAISS 索引和 Redis list 顺序完全相反

**修复**：
```python
pipe.rpush(self.name + "list", q)  # ✅ 从尾部插入，保持顺序一致
```

---

### 2. **严重 Bug：错误的索引映射**
```python
# call 方法中
filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
filtered_prompts = [pormpts[i] for i in filtered_ind]  # ❌ 错误！
```

**问题**：
- `filtered_ind` 存储的是枚举索引 `i`（在检索结果中的位置）
- 但应该使用 `ind[0][i]`（FAISS 中的真实索引）来映射到 Redis list

**修复**：
```python
filtered_ind = [(ind[0][i], d) for i, d in enumerate(dis[0]) if d < self.distance_threshold]
filtered_prompts = [prompts[idx] for idx, _ in filtered_ind]
```

---

### 3. **Bug：clear_cache 删除错误的 key**
```python
def clear_cache(self):
    pormpts = self.redis.lrange(self.name + "list", 0, -1)
    self.redis.delete(*pormpts)  # ❌ 删除的 key 不对
```

**问题**：
- 存储时的 key 是 `self.name + "key:" + q`
- 但这里直接删除 `pormpts`（prompt 内容本身）

**修复**：
```python
def clear_cache(self):
    prompts = self.redis.lrange(self.name + "list", 0, -1)
    keys_to_delete = [self.name + "key:" + q.decode() for q in prompts]
    if keys_to_delete:
        self.redis.delete(*keys_to_delete)
    self.redis.delete(self.name + "list")
    if os.path.exists(f"{self.name}.index"):
        os.unlink(f"{self.name}.index")
    self.index = None
```

---

### 4. **逻辑问题：过早返回**
```python
dis, ind = self.index.search(embedding, k=100)
if dis[0][0] > self.distance_threshold:  # 只检查第一个结果
    return None
# 后面又过滤所有结果...
```

**建议**：直接移除这个提前检查，让后续过滤处理所有情况。

---

### 5. **拼写错误**
```python
pormpts = ...  # ❌ 应该是 prompts
```

---

## 完整修复版本

```python
def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
    if isinstance(prompt, str):
        prompt = [prompt]
        response = [response]

    embedding = self.embedding_method(prompt)
    if self.index is None:
        self.index = faiss.IndexFlatL2(embedding.shape[1])

    self.index.add(embedding)
    faiss.write_index(self.index, f"{self.name}.index")

    try:
        with self.redis.pipeline() as pipe:
            for q, a in zip(prompt, response):
                pipe.setex(self.name + "key:" + q, self.ttl, a)
                pipe.rpush(self.name + "list", q)  # ✅ 修复：使用 rpush
            return pipe.execute()
    except:
        import traceback
        traceback.print_exc()
        return -1

def call(self, prompt: str):
    if self.index is None:
        return None

    embedding = self.embedding_method(prompt)
    dis, ind = self.index.search(embedding, k=100)
    
    # ✅ 修复：正确过滤和映射索引
    filtered_results = [(ind[0][i], dis[0][i]) for i in range(len(dis[0])) 
                        if dis[0][i] < self.distance_threshold]
    
    if not filtered_results:
        return None
    
    prompts = self.redis.lrange(self.name + "list", 0, -1)
    filtered_prompts = [prompts[idx] for idx, _ in filtered_results]
    
    return self.redis.mget([self.name + "key:" + q.decode() for q in filtered_prompts])

def clear_cache(self):
    prompts = self.redis.lrange(self.name + "list", 0, -1)
    if prompts:
        keys = [self.name + "key:" + q.decode() for q in prompts]
        self.redis.delete(*keys)
    self.redis.delete(self.name + "list")
    if os.path.exists(f"{self.name}.index"):
        os.unlink(f"{self.name}.index")
    self.index = None
```

这些 bug 会导致缓存检索完全错误，返回错误的答案或找不到已存储的内容。

# 问题一

```python
import numpy as np


def get_embedding(text):
    if isinstance(text, str):
        text = [text]
    return np.array([np.ones(768) for t in text])
```

**对问题进行编码，代码中直接返回的是全为1的ndarray，这会导致所有问题向量都是一模一样，
检索时所有得分全部一致，无法区分任何语义，检索结果完全随机，导致检索结果不准确**

# 问题二

```python
# embedding = self.embedding_method(prompt)
# if self.index is None:
#     self.index = faiss.IndexFlatL2(embedding.shape[1])
# self.index.add(embedding)
# faiss.write_index(self.index, f"{self.name}.index")

# with self.redis.pipeline() as pipe:
#     for q, a in zip(prompt, response):
#         pipe.setex(self.name + "key:" + q, self.ttl, a)  # 提问和回答存储在redis
#         pipe.lpush(self.name + "list", q)  # 所有的提问都存储在list里面，方便后续使用
# 
#     return pipe.execute()
```

```python
# 过滤不满足距离的结果
# filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
# 
# pormpts = self.redis.lrange(self.name + "list", 0, -1)
# print("pormpts", pormpts)
# filtered_prompts = [pormpts[i] for i in filtered_ind]
# 
# # 获取得到原始的提问 ，并在redis 找到对应的回答
# return self.redis.mget([self.name + "key:" + q.decode() for q in filtered_prompts])
```

**在store方法中，对原始问题编码后将其存入到XX.index文件中，每次执行前进行加载，
而后在call函数中通过faiss进行向量检索，然后返回检索到的向量在XX.index文件中的索引位置，这会导致：
一：如果存放在redis中的缓存被清除，此时self.redis.lrange获取到的数据与faiss检索到的索引不在一一匹配，导致filtered_prompts获取到的数据是混乱的，并非正确结果。
二：如果存放在redis中的缓存被清除，filtered_ind长度有可能大于pormpts，导致下标越界。
三：如果redis缓存未被清除，filtered_ind和pormpts长度一致，但是顺序却是相反的。
① pipe.lpush将q加入到双向链表中，从左侧一一加入，这会导致最先加入的在最右侧，当使用lrange获取双向链表中的数据时，frist_data在序列的最后面。
② self.index.add(embedding)在追加向量时是在末尾追加，如果此时检索到frist_data，他所对应的向量索引位置会在结果序列的首位
③ filtered_prompts = [pormpts[i] for i in filtered_ind] 此时再根据索引获取数据就会导致 filtered_ind和pormpts两者互不对应
**

# 问题三

```python
# embedding = self.embedding_method(prompt)
# if self.index is None:
#     self.index = faiss.IndexFlatL2(embedding.shape[1])
# 
# self.index.add(embedding)
# faiss.write_index(self.index, f"{self.name}.index")
```

**index.add追加时未做去重操作，会导致存在多个相同的向量，导致检索时返回的topK条结果都是一样的数据**

# 问题四

```python
# def clear_cache(self):
#     pormpts = self.redis.lrange(self.name + "list", 0, -1)
#     self.redis.delete(*pormpts)
#     self.redis.delete(self.name + "list")
#     os.unlink(f"{self.name}.index")
#     self.index = None
```

**self.redis.delete(pormpts)会删除以list表中的数据作为key的缓存数据
但是在存储时使用的key是pipe.setex(self.name + "key:" + q, self.ttl, a)
pormpts中的数据是q，需要拼接前缀才能删除以q作为key的redis缓存**

# 问题五

**使用faiss需要加载XX.index文件，将问题编码后写入到文件，对磁盘进行大量的读写操作，对内存不友好，若数据量很大效率也会降低。
改用向量数据库milvus或者文档数据库ElasticSearch效果更佳，也更加简单方便，同时逻辑更加清晰，返回信息更规范格式化。**

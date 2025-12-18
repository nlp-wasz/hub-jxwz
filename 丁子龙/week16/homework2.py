import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int=3600*24, # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        embedding = self.embedding_method(prompt)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        # 记录每条 prompt 对应的 FAISS ID（add 之前）
        faiss_ids = [self.index.ntotal + i for i in range(len(prompt))]

        # 添加向量到 FAISS
        self.index.add(embedding)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q, a, fid in zip(prompt, response, faiss_ids):
                    # 1️⃣ prompt -> response
                    pipe.setex(self.name + "key:" + q, self.ttl, a)
                    # 2️⃣ FAISS ID -> prompt 映射
                    pipe.hset(self.name + ":id2prompt", fid, q)
                    # 3️⃣ 可选：保留 list，用于遍历或调试
                    pipe.lpush(self.name + "list", q)

                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str):
        if self.index is None:
            return None

        # 新的提问进行编码
        embedding = self.embedding_method(prompt)

        # 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=100)
        if dis[0][0] > self.distance_threshold:
            return None

        # 过滤不满足距离的结果
        filtered_ids = [ind[0][i] for i, d in enumerate(dis[0]) if d < self.distance_threshold]
        # filtered_ind = [ind[0][i] for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        # pormpts = self.redis.lrange(self.name + "list", 0, -1)
        # print("pormpts", pormpts)
        # prompts = self.redis.
        # filtered_prompts = [pormpts[i] for i in filtered_ind]
        filtered_ids_str = [str(fid) for fid in filtered_ids]
        prompts = self.redis.hmget(self.name + ":id2prompt", filtered_ids_str)
        print("pormpts", prompts)

        responses = self.redis.mget([self.name + "key:" + p.decode() for p in prompts])
        # self.redis.mget([self.name + "key:" + q.decode() for q in filtered_prompts])

        # 获取得到原始的提问 ，并在redis 找到对应的回答
        return responses

    def clear_cache(self):
        pormpts = self.redis.lrange(self.name + "list", 0, -1)
        if pormpts:
            self.redis.delete(*pormpts)
            self.redis.delete(self.name + "list")
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
        self.index = None

if __name__ == "__main__":
    # def get_embedding(text):
    #     if isinstance(text, str):
    #         text = [text]
    #
    #     return np.array([np.ones(768) for t in text])


    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        embeddings = []
        for t in text:
            vec = np.zeros(3)
            vec[0] = len(t)  # 长度
            vec[1] = sum(map(ord, t)) % 10
            vec[2] = 1
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


    embed_cache = SemanticCache(
        name="semantic_ache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store("aaa", "A")
    embed_cache.store("bbb", "B")
    embed_cache.store("cccccccc", "C")
    embed_cache.store(prompt="hello world", response="hello world1232")
    print(embed_cache.call(prompt="hello world"))

    embed_cache.store(prompt="hello my bame", response="nihao")
    print(embed_cache.call(prompt="hello my bame"))

    embed_cache.store(prompt="my bame", response="bame")
    print(embed_cache.call(prompt="my bame"))



    # 这个 query 更接近 "cccccccc"
    print(embed_cache.call("cccccccc"))
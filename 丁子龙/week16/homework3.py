import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss
# from sentence_transformers import SentenceTransformer

from typing import Optional, List, Union, Any, Dict

class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
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
        # self.target_id = 0
        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

    def add_route(self, questions: List[str], target: str):
        if isinstance(questions, str):
            questions = [questions]
            # target = [target]

        embeddings = self.embedding_method(questions)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)  # L2 距离

        # 记录每条 prompt 对应的 FAISS ID（add 之前）
        start_id = self.index.ntotal  # FAISS 下一个插入 ID 就是当前总量
        faiss_ids = list(range(start_id, start_id + len(questions)))

        # 添加向量（FAISS 自动分配 ID）
        self.index.add(embeddings)
        faiss.write_index(self.index, f"{self.name}.index")


        try:
            with self.redis.pipeline() as pipe:
                for q, fid in zip(questions,  faiss_ids):
                    # 1️⃣ prompt -> response
                    pipe.setex(self.name + "key:" + q, self.ttl, target)
                    # 2️⃣ FAISS ID -> prompt 映射
                    pipe.hset(self.name + ":id2prompt", fid, q)
                    # 3️⃣ 可选：保留 list，用于遍历或调试
                    pipe.lpush(self.name + "list", q)
                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1


    def route(self, question: str):
        prompt = question
        if self.index is None:
            return None

        # 新的提问进行编码
        embedding = self.embedding_method(prompt)

        # 向量数据库中进行检索
        k = min(5, self.index.ntotal)
        dis, ind = self.index.search(embedding, k=k)
        if dis[0][0] > self.distance_threshold:
            return None

        # ✅ 关键修复：过滤距离 & 提取有效 IDs
        valid_indices = []
        for i in range(len(dis[0])):
            if dis[0][i] <= self.distance_threshold:
                valid_indices.append(ind[0][i])
            else:
                break  # 因 FAISS 返回升序，可提前终止
        if not valid_indices:
            return None

        valid_ids_str = [str(fid) for fid in valid_indices]
        prompts_bytes = self.redis.hmget(f"{self.name}:id2prompt", valid_ids_str)

        # 过滤掉 None（理论上不应出现，但防御性编程）
        prompts = [p.decode() for p in prompts_bytes if p is not None]
        if not prompts:
            return None

        # ✅ 关键修复：取第一个（最近邻）对应的 target 即可（业务上合理）
        # 若需聚合/投票，可扩展，但当前逻辑应返回最匹配的一个
        first_prompt = prompts[0]
        response = self.redis.get(f"{self.name}key:{first_prompt}")
        return response.decode() if response else None



if __name__ == "__main__":
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
    router = SemanticRouter(
        name="semantic_router",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    print(router.route("Hi, good morning"))
    print(router.route("如何退货"))
from typing import Optional, List, Dict
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util


class SemanticRouter:
    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            cache_ttl: int = 3600
    ):
        self.model = SentenceTransformer(model_name)
        self.cache_ttl = cache_ttl
        self.routes: Dict[str, List[np.ndarray]] = {}
        self.cache: Dict[str, str] = {}
        self.cache_timestamps: Dict[str, float] = {}

    def add_route(self, questions: List[str], target: str):
        # 编码为numpy数组（2维，shape: [len(questions), 嵌入维度]）
        embeddings = self.model.encode(questions, convert_to_tensor=False)
        if target not in self.routes:
            self.routes[target] = []
        # 每个句子的嵌入是1维数组，逐个添加到列表
        for emb in embeddings:
            self.routes[target].append(emb)

    def get_cached_result(self, question: str) -> Optional[str]:
        current_time = time.time()
        if question in self.cache:
            if current_time - self.cache_timestamps[question] < self.cache_ttl:
                print(f"命中缓存，返回目标：{self.cache[question]}")
                return self.cache[question]
            del self.cache[question]
            del self.cache_timestamps[question]
        return None

    def cache_result(self, question: str, result: str):
        self.cache[question] = result
        self.cache_timestamps[question] = time.time()

    def route(self, question: str, use_cache: bool = True) -> Optional[str]:
        if use_cache:
            cached_result = self.get_cached_result(question)
            if cached_result is not None:
                print(f"命中缓存，返回目标：{cached_result}")
                return cached_result

        if not self.routes:
            print("暂无路由配置")
            return None

        # 编码查询语句（返回2维数组，shape: [1, 嵌入维度]）
        question_embedding = self.model.encode(question, convert_to_tensor=False).reshape(1, -1)
        max_similarity = -1.0
        best_target = None

        for target, embeddings in self.routes.items():
            # 将目标下的所有嵌入拼接成2维数组（shape: [嵌入数量, 嵌入维度]）
            embeddings_batch = np.vstack(embeddings)
            # 计算余弦相似度（此时两个输入都是2维，格式匹配）
            similarities = util.cos_sim(question_embedding, embeddings_batch)
            avg_similarity = similarities.mean().item()
            print(f"目标 {target} 的平均相似度：{avg_similarity:.4f}")

            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                best_target = target

        if use_cache and best_target is not None:
            self.cache_result(question, best_target)
            print(f"缓存结果：{question} -> {best_target}")

        return best_target

    # 支持对象直接调用（可选）
    def __call__(self, question: str, use_cache: bool = True) -> Optional[str]:
        return self.route(question, use_cache)


if __name__ == "__main__":
    router = SemanticRouter()
    # 添加问候语路由
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello", "Good evening"],
        target="greeting"
    )
    # 添加退货路由
    router.add_route(
        questions=["如何退货", "退货流程是什么", "我想退货", "退货需要什么条件"],
        target="refund"
    )

    # 测试（支持直接调用对象）
    print("测试1：", router("Hi, good morning"))  # 返回 greeting
    print("测试2：", router("请问怎么退货？"))  # 返回 refund
    print("测试3：", router("Hello"))  # 命中缓存，返回 greeting
    print("测试4：", router("Good evening"))  # 返回 greeting
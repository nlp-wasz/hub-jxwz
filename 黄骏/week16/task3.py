from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRouter:
    """
    使用 TF-IDF + 余弦相似度进行意图识别路由
    """

    def __init__(self, default_threshold: float = 0.3):
        self.default_threshold = default_threshold
        self.routes: List[Dict[str, Any]] = []

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        self.reference_matrix = None
        self.reference_questions: List[str] = []
        self.reference_targets: List[str] = []
        self.reference_thresholds: List[float] = []

        self.cache: Dict[str, Dict[str, Any]] = {}

    def _rebuild_index(self):
        """路由更新后重建 TF-IDF 索引。"""
        self.reference_questions.clear()
        self.reference_targets.clear()
        self.reference_thresholds.clear()
        self.cache.clear()

        for route in self.routes:
            for q in route["questions"]:
                self.reference_questions.append(q)
                self.reference_targets.append(route["target"])
                self.reference_thresholds.append(route["threshold"])

        if self.reference_questions:
            self.reference_matrix = self.vectorizer.fit_transform(self.reference_questions)
        else:
            self.reference_matrix = None

    def add_route(self, questions: List[str], target: str, threshold: Optional[float] = None):
        if not questions:
            raise ValueError("questions 不能为空")

        route_threshold = threshold if threshold is not None else self.default_threshold
        self.routes.append(
            {"questions": questions, "target": target, "threshold": route_threshold}
        )
        self._rebuild_index()

    def route(self, question: str) -> Dict[str, Any]:
        if question in self.cache:
            return self.cache[question]

        if self.reference_matrix is None:
            raise ValueError("尚未添加路由。")

        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.reference_matrix)[0]

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        matched_question = self.reference_questions[best_idx]
        matched_target = self.reference_targets[best_idx] if best_score >= self.reference_thresholds[best_idx] else None

        result = {
            "question": question,
            "target": matched_target,
            "score": best_score,
            "matched_example": matched_question,
        }
        self.cache[question] = result
        return result

    def __call__(self, question: str) -> Dict[str, Any]:
        return self.route(question)


if __name__ == "__main__":
    router = SemanticRouter(default_threshold=0.25)
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货", "我要退款"],
        target="refund"
    )

    print(router("Hi, good morning"))
    print(router("我想退货"))
    # 重复调用走缓存
    print(router("Hi, good morning"))

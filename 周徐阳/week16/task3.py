import os
import numpy as np
from typing import Optional, List, Union, Any, Dict, Callable, Tuple
import faiss

class SemanticRouter:
    def __init__(
            self,
            name: str = "semantic_router",
            embedding_method: Callable[[Union[str, List[str]]], Any] = None,
            distance_threshold: float = 0.5,  # 相似度阈值
            aggregation: str = "max"  # 聚合方式：max, min, avg
    ):
        """
        语义路由器，用于快速实现意图识别
        
        Args:
            name: 路由器名称，用于持久化文件名
            embedding_method: 文本向量化方法
            distance_threshold: 距离阈值，超过此值返回 None（未匹配到意图）
            aggregation: 当一个意图有多个匹配结果时的聚合方式
        """
        self.name = name
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.aggregation = aggregation
        
        # 存储索引位置到意图的映射 {index: target}
        self.index_to_target: Dict[int, str] = {}
        
        # 存储意图到问题示例的映射 {target: [questions]}
        self.target_to_questions: Dict[str, List[str]] = {}
        
        # FAISS 索引
        self.index: Optional[faiss.Index] = None
        self.current_index_count = 0  # 当前索引数量
        
        # 加载已有的索引
        self._load_index()

    def add_route(self, questions: List[str], target: str):
        """
        添加路由规则
        
        Args:
            questions: 问题示例列表
            target: 目标意图标签
        """
        if not questions:
            raise ValueError("questions 不能为空")
        
        # 生成向量
        embeddings = self.embedding_method(questions)
        
        # 初始化或更新索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # 添加到 FAISS 索引
        start_idx = self.current_index_count
        self.index.add(embeddings)
        self.current_index_count += len(questions)
        
        # 更新映射关系
        for i, question in enumerate(questions):
            self.index_to_target[start_idx + i] = target
        
        # 更新意图到问题的映射
        if target not in self.target_to_questions:
            self.target_to_questions[target] = []
        self.target_to_questions[target].extend(questions)
        
        # 持久化
        self._save_index()

    def route(self, question: str, return_score: bool = False) -> Union[str, Tuple[str, float], None]:
        """
        路由问题到对应的意图
        
        Args:
            question: 待路由的问题
            return_score: 是否返回相似度分数
            
        Returns:
            如果 return_score=False: 返回意图标签或 None
            如果 return_score=True: 返回 (意图标签, 距离分数) 或 (None, None)
        """
        if self.index is None or self.current_index_count == 0:
            return (None, None) if return_score else None
        
        # 向量化查询
        embedding = self.embedding_method(question)
        
        # 搜索最相似的 k 个结果
        k = min(10, self.current_index_count)
        distances, indices = self.index.search(embedding, k=k)
        
        # 按意图聚合结果
        target_scores: Dict[str, List[float]] = {}
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if dist < self.distance_threshold:
                target = self.index_to_target.get(idx)
                if target:
                    if target not in target_scores:
                        target_scores[target] = []
                    target_scores[target].append(float(dist))
        
        # 没有匹配的意图
        if not target_scores:
            return (None, None) if return_score else None
        
        # 根据聚合策略选择最佳意图
        best_target = None
        best_score = float('inf')
        
        for target, scores in target_scores.items():
            if self.aggregation == "min":
                score = min(scores)
            elif self.aggregation == "max":
                score = max(scores)
            elif self.aggregation == "avg":
                score = sum(scores) / len(scores)
            else:
                score = min(scores)  # 默认使用最小距离
            
            if score < best_score:
                best_score = score
                best_target = target
        
        if return_score:
            return (best_target, best_score)
        return best_target

    def __call__(self, question: str, return_score: bool = False):
        """
        使路由器可直接调用
        """
        return self.route(question, return_score)

    def get_routes(self) -> Dict[str, List[str]]:
        """
        获取所有路由规则
        
        Returns:
            {意图: [问题示例列表]}
        """
        return self.target_to_questions.copy()

    def remove_route(self, target: str):
        """
        删除指定意图的路由规则
        注意：这需要重建整个索引
        """
        if target not in self.target_to_questions:
            return
        
        # 删除该意图
        del self.target_to_questions[target]
        
        # 重建索引
        self._rebuild_index()

    def clear(self):
        """
        清空所有路由规则
        """
        self.index = None
        self.index_to_target.clear()
        self.target_to_questions.clear()
        self.current_index_count = 0
        
        # 删除持久化文件
        index_file = f"{self.name}.index"
        mapping_file = f"{self.name}.mapping.npy"
        questions_file = f"{self.name}.questions.npy"
        
        for file in [index_file, mapping_file, questions_file]:
            if os.path.exists(file):
                os.unlink(file)

    def _rebuild_index(self):
        """
        重建索引（在删除路由后调用）
        """
        # 保存当前的意图和问题
        targets_questions = self.target_to_questions.copy()
        
        # 清空
        self.index = None
        self.index_to_target.clear()
        self.current_index_count = 0
        
        # 重新添加
        for target, questions in targets_questions.items():
            self.add_route(questions, target)

    def _save_index(self):
        """
        持久化索引和映射关系
        """
        if self.index is not None:
            faiss.write_index(self.index, f"{self.name}.index")
            
            # 保存映射关系
            np.save(f"{self.name}.mapping.npy", self.index_to_target)
            np.save(f"{self.name}.questions.npy", self.target_to_questions)

    def _load_index(self):
        """
        加载已有的索引和映射关系
        """
        index_file = f"{self.name}.index"
        mapping_file = f"{self.name}.mapping.npy"
        questions_file = f"{self.name}.questions.npy"
        
        if all(os.path.exists(f) for f in [index_file, mapping_file, questions_file]):
            self.index = faiss.read_index(index_file)
            self.index_to_target = np.load(mapping_file, allow_pickle=True).item()
            self.target_to_questions = np.load(questions_file, allow_pickle=True).item()
            self.current_index_count = self.index.ntotal


if __name__ == "__main__":
    # 模拟 embedding 方法
    def get_embedding(text):
        """
        这里用随机向量模拟，实际应使用真实的 embedding 模型
        如 OpenAI, sentence-transformers 等
        """
        if isinstance(text, str):
            text = [text]
        
        # 简单模拟：根据文本内容生成不同的向量
        embeddings = []
        for t in text:
            # 使用文本的 hash 作为种子，保证相同文本生成相同向量
            seed = hash(t) % (2**32)
            np.random.seed(seed)
            embeddings.append(np.random.randn(768))
        
        return np.array(embeddings, dtype='float32')

    # 创建路由器
    router = SemanticRouter(
        name="my_router",
        embedding_method=get_embedding,
        distance_threshold=100.0,  # 演示用，实际应根据模型调整
        aggregation="min"
    )
    
    # 清空之前的数据
    router.clear()
    
    # 添加路由规则
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello", "Hey there"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货", "退货流程是什么", "我想退款"],
        target="refund"
    )
    router.add_route(
        questions=["查询订单", "我的订单在哪里", "订单状态"],
        target="order_inquiry"
    )
    
    # 测试路由
    print("=" * 50)
    print("测试路由功能：")
    print("=" * 50)
    
    test_questions = [
        "Hi, good morning",
        "Hello there",
        "如何申请退款",
        "订单查询",
        "What's the weather like?"  # 不匹配任何意图
    ]
    
    for q in test_questions:
        result = router(q, return_score=True)
        print(f"问题: {q:30} -> 意图: {result[0]:20} (距离: {result[1]})")
    
    print("\n" + "=" * 50)
    print("所有路由规则：")
    print("=" * 50)
    routes = router.get_routes()
    for target, questions in routes.items():
        print(f"{target}:")
        for q in questions:
            print(f"  - {q}")

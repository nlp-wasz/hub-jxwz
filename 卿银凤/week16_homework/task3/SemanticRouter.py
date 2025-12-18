from typing import Optional, List, Union, Any, Dict, Callable
import numpy as np
import faiss
from openai import OpenAI

class SemanticRouter:
    def __init__(
            self,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            distance_threshold: float = 0.5
    ):
        """
        初始化语义路由器
        
        Args:
            embedding_method: 文本编码方法，将文本转换为向量
            distance_threshold: 距离阈值，超过此值则认为不匹配
        """
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.index = None
        self.routes = []  # 存储 (target, question) 的映射
        self.targets = []  # 存储每个向量对应的target

    def add_route(self, questions: List[str], target: str):
        """
        添加路由规则
        
        Args:
            questions: 该意图的示例问题列表
            target: 意图标签/目标
        """
        # 对问题进行编码
        embeddings = self.embedding_method(questions)
        
        # 初始化或更新FAISS索引
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # 添加向量到索引
        self.index.add(embeddings)
        
        # 记录每个问题对应的target
        for question in questions:
            self.routes.append((target, question))
            self.targets.append(target)

    def route(self, question: str) -> Optional[Dict[str, Any]]:
        """
        根据问题路由到对应的意图
        
        Args:
            question: 用户输入的问题
            
        Returns:
            包含target、matched_question和distance的字典，如果没有匹配则返回None
        """
        if self.index is None or self.index.ntotal == 0:
            return None
        
        # 对输入问题进行编码
        embedding = self.embedding_method(question)
        
        # 在索引中搜索最相似的向量
        distances, indices = self.index.search(embedding, k=1)
        
        # 检查是否有结果
        if len(distances[0]) == 0:
            return None
        
        best_distance = distances[0][0]
        best_index = indices[0][0]
        
        # 如果距离超过阈值，返回None
        if best_distance > self.distance_threshold:
            return {
                "target": None,
                "matched_question": None,
                "distance": float(best_distance),
                "message": "No matching route found"
            }
        
        # 返回匹配的target和相关信息
        target = self.targets[best_index]
        matched_question = self.routes[best_index][1]
        
        return {
            "target": target,
            "matched_question": matched_question,
            "distance": float(best_distance)
        }


if __name__ == "__main__":
    client = OpenAI(
    api_key="sk-9c40a7f2427c4c71af9e0f82fc906071", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 定义embedding方法
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=text
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype='float32')
    
    # 创建路由器
    router = SemanticRouter(
        embedding_method=get_embedding,
        distance_threshold=0.5
    )
    
    # 添加路由规则
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello", "Hey there"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货", "我想退货", "退货流程是什么"],
        target="refund"
    )
    router.add_route(
        questions=["订单在哪里", "查询订单", "我的订单状态"],
        target="order_query"
    )
    
    # 测试路由
    test_questions = [
        "Hi, good morning",
        "如何退货",
        "你好",
        "我想查询订单"
    ]
    
    for q in test_questions:
        result = router.route(q)
        print(f"问题: {q}")
        print(f"结果: {result}")
        print("-" * 50)
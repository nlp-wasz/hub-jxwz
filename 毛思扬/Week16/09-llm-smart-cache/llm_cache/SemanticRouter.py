import redis
from typing import Optional, List, Union, Any, Dict


class SemanticRouter:
    def __init__(
            self,
            name: str,
            redis_url: str = "127.0.0.1",
            redis_port: int = 6379,
            redis_password: str = "123456",
            distance_threshold: float = 0.7
    ):
        """
        初始化语义路由器
        
        Args:
            name: 路由器名称
            redis_url: Redis服务器地址
            redis_port: Redis端口
            redis_password: Redis密码
            distance_threshold: 相似度阈值
        """
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.distance_threshold = distance_threshold
        self.routes = {}  # 存储路由信息

    def add_route(self, questions: List[str], target: str, metadata: Optional[Dict] = None):
        """
        添加路由规则
        
        Args:
            questions: 示例问题列表
            target: 目标路由
            metadata: 元数据
        """
        # 将路由信息存储到Redis中
        route_key = f"route:{self.name}:{target}"
        route_data = {
            "target": target,
            "questions": questions,
            "metadata": metadata or {}
        }

        # 存储路由信息
        self.redis.hset(route_key, mapping={
            "target": target,
            "metadata": str(metadata) if metadata else "{}",
            "questions": str(questions)
        })

        # 将路由目标添加到路由列表中
        self.redis.sadd(f"routes:{self.name}", target)

        # 缓存路由信息到内存
        self.routes[target] = route_data

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的相似度（这里使用简化的字符级相似度）
        在实际应用中，应使用真实的嵌入模型计算余弦相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        # 这里只是一个占位符实现，实际应该使用嵌入模型
        # 比如使用 sentence-transformers 或其他嵌入模型计算真实相似度
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def route(self, question: str) -> Optional[Dict[str, Any]]:
        """
        根据问题内容路由到相应的目标
        
        Args:
            question: 输入问题
            
        Returns:
            路由结果，包含目标和元数据
        """
        # 检查是否有缓存的结果
        cached_result = self.redis.hgetall(f"route_cache:{self.name}:{question}")
        if cached_result:
            return {
                "target": cached_result["target"],
                "metadata": eval(cached_result["metadata"]),
                "similarity": float(cached_result["similarity"])
            }

        best_match = None
        best_similarity = 0.0
        best_metadata = {}

        # 获取所有路由
        routes = self.redis.smembers(f"routes:{self.name}")

        # 遍历所有路由寻找最佳匹配
        for target in routes:
            route_key = f"route:{self.name}:{target}"
            route_info = self.redis.hgetall(route_key)

            if not route_info:
                continue

            questions = eval(route_info["questions"])
            metadata = eval(route_info["metadata"]) if route_info.get("metadata") else {}

            # 计算与所有示例问题的相似度
            for ref_question in questions:
                similarity = self._calculate_similarity(question, ref_question)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target
                    best_metadata = metadata

        # 如果最佳匹配的相似度高于阈值，则返回结果
        if best_similarity >= self.distance_threshold and best_match:
            result = {
                "target": best_match,
                "metadata": best_metadata,
                "similarity": best_similarity
            }

            # 缓存结果
            self.redis.hset(f"route_cache:{self.name}:{question}", mapping={
                "target": best_match,
                "metadata": str(best_metadata),
                "similarity": str(best_similarity)
            })
            self.redis.expire(f"route_cache:{self.name}:{question}", 3600)  # 缓存1小时

            return result

        # 没有找到合适的路由
        return None

    def clear_routes(self):
        """清除所有路由"""
        routes = self.redis.smembers(f"routes:{self.name}")
        for target in routes:
            self.redis.delete(f"route:{self.name}:{target}")
        self.redis.delete(f"routes:{self.name}")
        self.routes.clear()

    def get_routes(self) -> List[str]:
        """获取所有路由目标"""
        return list(self.redis.smembers(f"routes:{self.name}"))


if __name__ == "__main__":
    # 示例使用
    router = SemanticRouter(name="topic-router", distance_threshold=0.3)

    # 清除之前的路由（测试用）
    router.clear_routes()

    # 添加问候路由
    router.add_route(
        questions=["hello", "hi", "good morning", "good afternoon"],
        target="greeting",
        metadata={"type": "greeting"}
    )

    # 添加告别路由
    router.add_route(
        questions=["bye", "goodbye", "see you later"],
        target="farewell",
        metadata={"type": "farewell"}
    )

    # 添加退款路由
    router.add_route(
        questions=["如何退货", "怎么退款", "退款流程"],
        target="refund",
        metadata={"type": "customer_service", "category": "refund"}
    )

    # 测试路由
    print("路由测试:")
    test_questions = ["Hi there!", "再见", "我想退货"]

    for q in test_questions:
        result = router.route(q)
        if result:
            print(f"问题: '{q}' -> 目标: {result['target']}, 相似度: {result['similarity']:.2f}")
        else:
            print(f"问题: '{q}' -> 未找到匹配路由")

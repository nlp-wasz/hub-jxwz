import redis
import json
import numpy as np
from typing import Optional, List, Union, Any, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib

# 向量生成器接口
class Vectorizer:
    """向量生成器抽象类"""
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """将文本转换为向量"""
        raise NotImplementedError

class RouteMatchStrategy(Enum):
    """路由匹配策略"""
    EXACT = "exact"           # 精确匹配
    SEMANTIC = "semantic"     # 语义匹配
    KEYWORD = "keyword"       # 关键词匹配

@dataclass
class Route:
    """路由定义"""
    id: str
    name: str
    target: str
    questions: List[str]
    embeddings: Optional[np.ndarray] = None
    strategy: RouteMatchStrategy = RouteMatchStrategy.SEMANTIC
    threshold: float = 0.8    # 语义相似度阈值
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SemanticRouter:
    def __init__(
        self,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        vectorizer: Optional[Vectorizer] = None,
        namespace: str = "semantic_router"
    ):
        """
        初始化语义路由器
        
        Args:
            redis_url: Redis 服务器地址
            redis_port: Redis 端口
            redis_password: Redis 密码
            vectorizer: 向量生成器
            namespace: Redis 键名前缀
        """
        # Redis 连接
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        
        self.namespace = namespace
        self.vectorizer = vectorizer
        
        # 路由缓存
        self.routes: Dict[str, Route] = {}
        
        # 初始化 Redis 数据结构
        self._init_redis_structure()
    
    def _init_redis_structure(self):
        """初始化 Redis 数据结构"""
        # 使用 Set 存储所有路由ID
        self.routes_key = f"{self.namespace}:routes"
        
        # 使用 Hash 存储路由元数据
        self.metadata_key = f"{self.namespace}:route_metadata"
        
        # 使用 Sorted Set 存储向量索引（用于快速查找）
        self.vector_index_key = f"{self.namespace}:vector_index"
        
        # 使用 Hash 存储精确匹配的映射
        self.exact_match_key = f"{self.namespace}:exact_matches"
        
        print(f"✅ SemanticRouter 初始化完成，命名空间: {self.namespace}")
    
    def _generate_route_id(self, target: str) -> str:
        """生成路由ID"""
        # 使用目标名 + 时间戳生成唯一ID
        import time
        timestamp = int(time.time() * 1000)
        return f"{target}_{timestamp}"
    
    def add_route(
        self,
        questions: List[str],
        target: str,
        route_name: Optional[str] = None,
        strategy: RouteMatchStrategy = RouteMatchStrategy.SEMANTIC,
        threshold: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加路由规则
        
        Args:
            questions: 触发该路由的问题列表
            target: 路由目标（可以是函数名、URL、处理程序等）
            route_name: 路由名称
            strategy: 匹配策略
            threshold: 语义匹配阈值
            metadata: 路由元数据
            
        Returns:
            路由ID
        """
        # 生成路由ID
        route_id = self._generate_route_id(target)
        route_name = route_name or f"route_{route_id}"
        
        # 生成向量嵌入（如果使用语义匹配且有向量生成器）
        embeddings = None
        if strategy == RouteMatchStrategy.SEMANTIC and self.vectorizer:
            embeddings = self.vectorizer.embed(questions)
        
        # 创建路由对象
        route = Route(
            id=route_id,
            name=route_name,
            target=target,
            questions=questions,
            embeddings=embeddings,
            strategy=strategy,
            threshold=threshold,
            metadata=metadata or {}
        )
        
        # 保存到 Redis
        self._save_route_to_redis(route)
        
        # 缓存到内存
        self.routes[route_id] = route
        
        print(f"✅ 路由添加成功: {route_name} -> {target}")
        print(f"   问题数量: {len(questions)}, 策略: {strategy.value}")
        
        return route_id
    
    def _save_route_to_redis(self, route: Route):
        """将路由保存到 Redis"""
        # 使用 pipeline 批量操作
        with self.redis.pipeline() as pipe:
            # 1. 添加到路由集合
            pipe.sadd(self.routes_key, route.id)
            
            # 2. 存储路由元数据
            route_data = {
                "id": route.id,
                "name": route.name,
                "target": route.target,
                "questions": json.dumps(route.questions, ensure_ascii=False),
                "strategy": route.strategy.value,
                "threshold": str(route.threshold),
                "metadata": json.dumps(route.metadata, ensure_ascii=False)
            }
            pipe.hset(self.metadata_key, route.id, json.dumps(route_data))
            
            # 3. 根据策略存储不同的索引
            if route.strategy == RouteMatchStrategy.EXACT:
                # 精确匹配：存储问题到路由ID的映射
                for question in route.questions:
                    normalized_q = question.lower().strip()
                    pipe.hset(self.exact_match_key, normalized_q, route.id)
            
            elif route.strategy == RouteMatchStrategy.SEMANTIC and route.embeddings is not None:
                # 语义匹配：存储向量索引
                # 这里简化处理，实际应该使用向量数据库
                # 为每个问题生成一个向量签名
                for i, question in enumerate(route.questions):
                    # 生成问题的向量签名（简化版：使用哈希）
                    vector_signature = self._generate_vector_signature(question)
                    # 使用 Sorted Set 存储，分数为路由ID的哈希值
                    score = int(hashlib.md5(route.id.encode()).hexdigest()[:8], 16)
                    pipe.zadd(self.vector_index_key, {vector_signature: score})
                    # 存储向量签名到路由的映射
                    pipe.hset(f"{self.namespace}:vector_map:{vector_signature}", 
                             "route_id", route.id)
                    pipe.hset(f"{self.namespace}:vector_map:{vector_signature}",
                             "question_index", str(i))
            
            elif route.strategy == RouteMatchStrategy.KEYWORD:
                # 关键词匹配：存储关键词索引
                for question in route.questions:
                    # 提取关键词（这里简单分割）
                    keywords = question.lower().split()
                    for keyword in keywords:
                        if len(keyword) > 2:  # 忽略太短的关键词
                            pipe.sadd(f"{self.namespace}:keyword:{keyword}", route.id)
            
            pipe.execute()
    
    def _generate_vector_signature(self, text: str) -> str:
        """生成向量的签名（简化版，实际应该使用真实的向量）"""
        # 这里使用文本哈希作为向量签名的简化表示
        # 实际应用中应该使用真实的向量和向量数据库
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简化版）"""
        # 这里使用简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def route(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        路由查询
        
        Args:
            question: 用户问题
            top_k: 返回前K个结果
            min_score: 最小匹配分数
            
        Returns:
            匹配的路由列表，按匹配度排序
        """
        print(f"\n🔍 路由查询: '{question}'")
        
        # 获取所有路由ID
        route_ids = self.redis.smembers(self.routes_key)
        
        if not route_ids:
            print("⚠️ 没有可用的路由规则")
            return []
        
        # 收集匹配结果
        matches = []
        
        for route_id in route_ids:
            # 获取路由信息
            route_data = self.redis.hget(self.metadata_key, route_id)
            if not route_data:
                continue
            
            route_dict = json.loads(route_data)
            route_strategy = RouteMatchStrategy(route_dict["strategy"])
            
            # 根据策略进行匹配
            score = 0.0
            matched_question = None
            
            if route_strategy == RouteMatchStrategy.EXACT:
                # 精确匹配
                normalized_q = question.lower().strip()
                matched_route_id = self.redis.hget(self.exact_match_key, normalized_q)
                if matched_route_id == route_id:
                    score = 1.0
                    matched_question = question
            
            elif route_strategy == RouteMatchStrategy.SEMANTIC:
                # 语义匹配
                questions = json.loads(route_dict["questions"])
                
                # 计算与每个问题的相似度
                max_similarity = 0.0
                best_question = None
                
                for q in questions:
                    similarity = self._calculate_similarity(question, q)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_question = q
                
                score = max_similarity
                matched_question = best_question
            
            elif route_strategy == RouteMatchStrategy.KEYWORD:
                # 关键词匹配
                questions = json.loads(route_dict["questions"])
                question_keywords = set(question.lower().split())
                
                max_keyword_score = 0.0
                best_question = None
                
                for q in questions:
                    route_keywords = set(q.lower().split())
                    if len(route_keywords) == 0:
                        continue
                    
                    # 计算关键词匹配度
                    common_keywords = question_keywords.intersection(route_keywords)
                    keyword_score = len(common_keywords) / len(route_keywords)
                    
                    if keyword_score > max_keyword_score:
                        max_keyword_score = keyword_score
                        best_question = q
                
                score = max_keyword_score
                matched_question = best_question
            
            # 检查是否达到阈值
            threshold = float(route_dict.get("threshold", 0.8))
            if score >= threshold and score >= min_score:
                matches.append({
                    "route_id": route_id,
                    "route_name": route_dict["name"],
                    "target": route_dict["target"],
                    "score": score,
                    "matched_question": matched_question,
                    "strategy": route_strategy.value,
                    "metadata": json.loads(route_dict.get("metadata", "{}"))
                })
        
        # 按分数排序
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回前K个结果
        result = matches[:top_k]
        
        if result:
            print(f"✅ 找到 {len(result)} 个匹配路由:")
            for i, match in enumerate(result):
                print(f"  {i+1}. [{match['strategy']}] {match['route_name']} -> {match['target']}")
                print(f"     匹配问题: {match['matched_question']}")
                print(f"     相似度: {match['score']:.3f}")
        else:
            print("❌ 没有匹配的路由")
        
        return result
    
    def get_route(self, question: str) -> Optional[str]:
        """
        获取最匹配的路由目标（简化接口）
        
        Args:
            question: 用户问题
            
        Returns:
            最匹配的路由目标，如果没有匹配则返回 None
        """
        matches = self.route(question, top_k=1)
        if matches:
            return matches[0]["target"]
        return None
    
    def __call__(self, question: str) -> Optional[str]:
        """使路由器可调用"""
        return self.get_route(question)
    
    def list_routes(self) -> List[Dict[str, Any]]:
        """列出所有路由"""
        route_ids = self.redis.smembers(self.routes_key)
        routes = []
        
        for route_id in route_ids:
            route_data = self.redis.hget(self.metadata_key, route_id)
            if route_data:
                routes.append(json.loads(route_data))
        
        return routes
    
    def delete_route(self, route_id: str) -> bool:
        """删除路由"""
        # 获取路由信息
        route_data = self.redis.hget(self.metadata_key, route_id)
        if not route_data:
            return False
        
        route_dict = json.loads(route_data)
        route_strategy = RouteMatchStrategy(route_dict["strategy"])
        
        with self.redis.pipeline() as pipe:
            # 从路由集合中移除
            pipe.srem(self.routes_key, route_id)
            
            # 删除路由元数据
            pipe.hdel(self.metadata_key, route_id)
            
            # 根据策略删除索引
            if route_strategy == RouteMatchStrategy.EXACT:
                # 删除精确匹配索引
                questions = json.loads(route_dict["questions"])
                for question in questions:
                    normalized_q = question.lower().strip()
                    pipe.hdel(self.exact_match_key, normalized_q)
            
            elif route_strategy == RouteMatchStrategy.SEMANTIC:
                # 删除语义索引（简化处理）
                questions = json.loads(route_dict["questions"])
                for question in questions:
                    vector_signature = self._generate_vector_signature(question)
                    pipe.zrem(self.vector_index_key, vector_signature)
                    pipe.delete(f"{self.namespace}:vector_map:{vector_signature}")
            
            elif route_strategy == RouteMatchStrategy.KEYWORD:
                # 删除关键词索引
                questions = json.loads(route_dict["questions"])
                for question in questions:
                    keywords = question.lower().split()
                    for keyword in keywords:
                        if len(keyword) > 2:
                            pipe.srem(f"{self.namespace}:keyword:{keyword}", route_id)
            
            pipe.execute()
        
        # 从内存缓存中移除
        if route_id in self.routes:
            del self.routes[route_id]
        
        print(f"🗑️ 路由删除成功: {route_id}")
        return True
    
    def clear_all_routes(self):
        """清除所有路由"""
        route_ids = self.redis.smembers(self.routes_key)
        
        for route_id in route_ids:
            self.delete_route(route_id)
        
        print(f"🧹 已清除所有路由，共 {len(route_ids)} 个")

# 简单的向量生成器实现
class SimpleVectorizer(Vectorizer):
    """简单的向量生成器（用于演示）"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """生成伪随机向量（实际应使用BERT等模型）"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = []
        for t in text:
            # 使用文本哈希作为随机种子，确保相同文本生成相同向量
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)
            vec = np.random.randn(self.dimension).astype('float32')
            # 归一化
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        
        return np.array(embeddings)

# 使用示例
if __name__ == "__main__":
    # 创建简单的向量生成器
    vectorizer = SimpleVectorizer(dimension=768)
    
    # 创建语义路由器
    router = SemanticRouter(
        namespace="demo_router",
        vectorizer=vectorizer
    )
    
    # 添加路由规则
    print("=== 添加路由规则 ===")
    
    # 问候路由
    greeting_route_id = router.add_route(
        questions=[
            "Hi, good morning",
            "Hi, good afternoon", 
            "Hello there",
            "Good evening",
            "Hey, how are you?",
            "早上好",
            "下午好",
            "晚上好"
        ],
        target="greeting_handler",
        route_name="问候路由",
        strategy=RouteMatchStrategy.SEMANTIC,
        threshold=0.6
    )
    
    # 退货路由
    refund_route_id = router.add_route(
        questions=[
            "如何退货",
            "怎么办理退货",
            "退货流程是什么",
            "我想退货",
            "退货需要什么条件",
            "return policy",
            "how to return items",
            "refund process"
        ],
        target="refund_handler",
        route_name="退货路由",
        strategy=RouteMatchStrategy.SEMANTIC,
        threshold=0.7,
        metadata={"category": "customer_service", "priority": "high"}
    )
    
    # 精确匹配路由
    exact_route_id = router.add_route(
        questions=[
            "订单状态",
            "查看物流",
            "track order"
        ],
        target="order_status_handler",
        route_name="订单状态路由",
        strategy=RouteMatchStrategy.EXACT
    )
    
    # 关键词匹配路由
    keyword_route_id = router.add_route(
        questions=[
            "产品咨询",
            "商品信息",
            "product information"
        ],
        target="product_info_handler",
        route_name="产品信息路由",
        strategy=RouteMatchStrategy.KEYWORD
    )
    
    print(f"\n=== 列出所有路由 ===")
    routes = router.list_routes()
    print(f"总路由数: {len(routes)}")
    
    print("\n=== 测试路由功能 ===")
    
    # 测试用例
    test_cases = [
        "Hi, good morning",  # 应该匹配问候路由
        "如何退货",           # 应该匹配退货路由
        "订单状态",           # 应该精确匹配
        "产品咨询",           # 应该关键词匹配
        "你好，世界",         # 可能不匹配或低分匹配
        "我想知道怎么退货商品",  # 应该匹配退货路由
        "Good evening everyone",  # 应该匹配问候路由
        "物流查询",           # 可能不匹配
        "商品退货政策",        # 应该匹配退货路由
        "hello",             # 应该匹配问候路由
    ]
    
    for test_question in test_cases:
        print(f"\n测试问题: '{test_question}'")
        result = router(test_question)
        if result:
            print(f"  路由到: {result}")
        else:
            print("  没有匹配的路由")
    
    print(f"\n=== 详细路由查询示例 ===")
    detailed_results = router.route("我想退货商品", top_k=2)
    for result in detailed_results:
        print(f"  匹配: {result['route_name']} -> {result['target']} (分数: {result['score']:.3f})")
    
    print(f"\n=== 删除路由示例 ===")
    router.delete_route(keyword_route_id)
    
    print(f"\n=== 清理所有路由 ===")
    router.clear_all_routes()
    
    # 测试清理后
    print(f"\n=== 清理后查询测试 ===")
    result = router("如何退货")
    print(f"路由结果: {result}")
```python
import numpy as np
import redis
from typing import Optional, List, Union, Any, Dict, Tuple
import hashlib
import json
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class Route:
    def __init__(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict] = None,
        distance_threshold: float = 0.3
    ):
        self.name = name
        self.references = references
        self.metadata = metadata or {}
        self.distance_threshold = distance_threshold
        self.reference_embeddings = None

class SemanticRouter:
    def __init__(
        self,
        name: str,
        embedding_method: callable,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        default_threshold: float = 0.3
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.embedding_method = embedding_method
        self.default_threshold = default_threshold
        self.routes = {}
        self.route_index = None
        
        # 从Redis加载已有的路由配置
        self._load_routes_from_redis()
        
    def _load_routes_from_redis(self):
        """从Redis加载路由配置"""
        try:
            routes_data = self.redis.get(f"{self.name}:routes")
            if routes_data:
                routes_dict = json.loads(routes_data)
                for route_name, route_data in routes_dict.items():
                    route = Route(
                        name=route_name,
                        references=route_data['references'],
                        metadata=route_data.get('metadata', {}),
                        distance_threshold=route_data.get('distance_threshold', self.default_threshold)
                    )
                    self.routes[route_name] = route
                
                # 重建向量索引
                self._build_index()
        except Exception as e:
            print(f"加载路由配置失败: {e}")
    
    def _save_routes_to_redis(self):
        """保存路由配置到Redis"""
        try:
            routes_dict = {}
            for route_name, route in self.routes.items():
                routes_dict[route_name] = {
                    'references': route.references,
                    'metadata': route.metadata,
                    'distance_threshold': route.distance_threshold
                }
            self.redis.set(f"{self.name}:routes", json.dumps(routes_dict))
        except Exception as e:
            print(f"保存路由配置失败: {e}")
    
    def _build_index(self):
        """构建路由的向量索引"""
        if not self.routes:
            self.route_index = None
            return
            
        # 收集所有路由的参考向量
        all_embeddings = []
        route_mapping = []  # 记录每个向量对应的路由
        
        for route_name, route in self.routes.items():
            if route.reference_embeddings is None:
                # 首次使用，生成参考向量的嵌入
                route.reference_embeddings = self.embedding_method(route.references)
            
            for embedding in route.reference_embeddings:
                all_embeddings.append(embedding)
                route_mapping.append(route_name)
        
        if all_embeddings:
            # 使用Faiss构建索引
            dim = len(all_embeddings[0])
            self.route_index = faiss.IndexFlatIP(dim)  # 使用内积（余弦相似度）
            
            # 归一化向量以便使用内积计算余弦相似度
            embeddings_array = np.array(all_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.route_index.add(embeddings_array)
            
            self.route_mapping = route_mapping
    
    def add_route(self, route: Route):
        """添加路由"""
        self.routes[route.name] = route
        self._save_routes_to_redis()
        self._build_index()
    
    def create_route(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict] = None,
        distance_threshold: Optional[float] = None
    ):
        """创建并添加新路由"""
        route = Route(
            name=name,
            references=references,
            metadata=metadata,
            distance_threshold=distance_threshold or self.default_threshold
        )
        self.add_route(route)
        return route
    
    def remove_route(self, route_name: str):
        """移除路由"""
        if route_name in self.routes:
            del self.routes[route_name]
            self._save_routes_to_redis()
            self._build_index()
            return True
        return False
    
    def __call__(self, query: str, top_k: int = 1) -> List[Tuple[Route, float]]:
        """路由查询主方法"""
        if not self.routes or self.route_index is None:
            return []
        
        # 获取查询的嵌入向量
        query_embedding = self.embedding_method([query])[0]
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # 搜索最相似的路由参考
        similarities, indices = self.route_index.search(query_vector, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            similarity = similarities[0][i]
            route_name = self.route_mapping[idx]
            route = self.routes[route_name]
            
            # 检查是否超过路由的阈值
            if similarity >= (1 - route.distance_threshold):
                results.append((route, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def route(self, query: str) -> Optional[Route]:
        """路由查询，返回最佳匹配的路由"""
        results = self(query, top_k=1)
        return results[0][0] if results else None
    
    def get_best_match(self, query: str) -> Optional[Tuple[Route, float]]:
        """获取最佳匹配的路由及其相似度"""
        results = self(query, top_k=1)
        return results[0] if results else None
    
    def clear_all_routes(self):
        """清空所有路由"""
        self.routes = {}
        self.route_index = None
        self.redis.delete(f"{self.name}:routes")

# 使用示例
if __name__ == "__main__":
    # 伪代码：嵌入生成函数
    def get_embedding(texts: List[str]) -> List[List[float]]:
        """伪代码：实际使用时替换为真实的嵌入模型"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 这里应该是真实的嵌入生成逻辑，如调用OpenAI、Cohere等API
        # 返回形状为 [len(texts), embedding_dim] 的列表
        return [list(np.random.rand(768)) for _ in texts]
    
    # 创建语义路由器
    router = SemanticRouter(
        name="my-router",
        embedding_method=get_embedding
    )
    
    # 创建路由
    greeting_route = router.create_route(
        name="greeting",
        references=["hello", "hi", "good morning", "good afternoon"],
        metadata={"type": "greeting", "response_template": "Hello! How can I help you?"},
        distance_threshold=0.2
    )
    
    farewell_route = router.create_route(
        name="farewell", 
        references=["bye", "goodbye", "see you", "farewell"],
        metadata={"type": "farewell", "response_template": "Goodbye! Have a great day!"},
        distance_threshold=0.2
    )
    
    refund_route = router.create_route(
        name="refund",
        references=["how to return", "refund policy", "return item", "get money back"],
```
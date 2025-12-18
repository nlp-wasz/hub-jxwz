import numpy as np
import redis
import json
import hashlib
from typing import Optional, List, Union, Any, Dict, Callable
import os
import faiss


class SemanticRouter:
    def __init__(
            self,
            name: str = "semantic_router",
            ttl: int = 3600*24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            embedding_method: Callable = None,
            distance_threshold: float = 0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True  # 使用字符串解码
        )
        self.ttl = ttl
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold

        # 初始化FAISS索引
        index_path = f"{self.name}.index"
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"Failed to load index: {e}")
                self.index = None
        else:
            self.index = None
            
        # 路由映射存储在Redis中
        self.routes_key = f"{self.name}:routes"
        
        # 确保索引和数据库的一致性
        if self.index is not None:
            # 清理可能存在的孤立数据
            self._sync_index_with_redis()

    def _sync_index_with_redis(self):
        """同步索引和Redis数据"""
        try:
            # 获取Redis中存储的所有问题
            all_question_hashes = self.redis.lrange(self.routes_key, 0, -1)
            if len(all_question_hashes) != self.index.ntotal:
                print(f"Index and Redis data mismatch. Index: {self.index.ntotal}, Redis: {len(all_question_hashes)}")
        except Exception as e:
            print(f"Sync error: {e}")

    def add_route(self, questions: List[str], target: str):
        """
        添加路由规则
        """
        if not questions or not target:
            return False
            
        try:
            # 为每个问题创建嵌入向量
            embeddings = self.embedding_method(questions)
            
            # 初始化索引（如果尚未初始化）
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                
            # 记录添加前的索引总数，用于确定新添加向量的索引位置
            start_index = self.index.ntotal if self.index else 0
            
            # 添加嵌入向量到索引
            self.index.add(embeddings)
            faiss.write_index(self.index, f"{self.name}.index")
            
            # 在Redis中存储问题到目标的映射
            with self.redis.pipeline() as pipe:
                for i, question in enumerate(questions):
                    # 使用问题的哈希值作为唯一标识
                    question_hash = hashlib.md5(question.encode()).hexdigest()
                    key = f"{self.name}:question:{question_hash}"
                    
                    # 存储问题和目标的映射关系
                    route_data = {
                        "question": question,
                        "target": target,
                        "embedding_index": int(start_index + i)  # 确保是整数类型
                    }
                    pipe.setex(key, self.ttl, json.dumps(route_data))
                    
                    # 检查是否已经存在于列表中，避免重复
                    if not self.redis.lrem(self.routes_key, 0, question_hash):
                        # 将问题哈希值添加到路由列表中
                        pipe.lpush(self.routes_key, question_hash)
                
                pipe.execute()
                
            return True
            
        except Exception as e:
            print(f"Error adding route: {e}")
            return False

    def route(self, question: str):
        """
        根据问题找到对应的路由目标
        """
        if not question:
            return None
            
        try:
            # 首先检查是否有精确匹配的缓存
            question_hash = hashlib.md5(question.encode()).hexdigest()
            cached_result = self.redis.get(f"{self.name}:route_cache:{question_hash}")
            if cached_result:
                return json.loads(cached_result)
                
            # 如果没有缓存，则进行语义搜索
            if self.index is None:
                print("Index is None")
                return None
                
            # 生成问题的嵌入向量
            embedding = self.embedding_method([question])
            
            # 使用FAISS进行相似性搜索
            distances, indices = self.index.search(embedding, k=5)
            print(f"Search results - Distances: {distances[0]}, Indices: {indices[0]}")
            
            # 检查最近的匹配是否在阈值范围内
            for i in range(len(distances[0])):
                distance = float(distances[0][i])
                if distance <= self.distance_threshold:
                    # 找到最相似的问题对应的路由
                    nearest_index = int(indices[0][i])
                    print(f"Found match at index {nearest_index} with distance {distance}")
                    
                    # 遍历所有问题查找索引匹配的问题
                    all_question_hashes = self.redis.lrange(self.routes_key, 0, -1)
                    print(f"Total questions in Redis: {len(all_question_hashes)}")
                    
                    # 查找索引匹配的问题
                    for hash_str in all_question_hashes:
                        route_key = f"{self.name}:question:{hash_str}"
                        route_data = self.redis.get(route_key)
                        
                        if route_data:
                            try:
                                route_info = json.loads(route_data)
                                if route_info.get("embedding_index") == nearest_index:
                                    target = route_info["target"]
                                    print(f"Found target: {target}")
                                    
                                    # 缓存结果
                                    self.redis.setex(
                                        f"{self.name}:route_cache:{question_hash}", 
                                        self.ttl, 
                                        json.dumps(target)
                                    )
                                    
                                    return target
                            except json.JSONDecodeError:
                                print(f"Failed to decode route data for key: {route_key}")
                                continue
                
            print("No matching route found")
            return None
            
        except Exception as e:
            print(f"Error in routing: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def clear_routes(self):
        """
        清除所有路由规则
        """
        try:
            # 删除所有路由相关数据
            route_keys = self.redis.keys(f"{self.name}:question:*")
            route_cache_keys = self.redis.keys(f"{self.name}:route_cache:*")
            
            if route_keys:
                self.redis.delete(*route_keys)
                
            if route_cache_keys:
                self.redis.delete(*route_cache_keys)
                
            self.redis.delete(self.routes_key)
            
            # 删除FAISS索引文件
            index_path = f"{self.name}.index"
            if os.path.exists(index_path):
                os.unlink(index_path)
                
            self.index = None
            return True
        except Exception as e:
            print(f"Error clearing routes: {e}")
            return False


if __name__ == "__main__":
    # 示例用法 - 使用固定的向量以确保可重现的结果
    def mock_embedding_method(texts):
        """模拟嵌入方法 - 使用固定向量确保一致性"""
        if isinstance(texts, str):
            texts = [texts]
            
        # 为确保一致性，我们根据文本内容生成确定性的向量
        embeddings = []
        for text in texts:
            # 使用文本的哈希值生成确定性的向量
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val % (2**32))  # 使用哈希值作为种子
            embedding = np.random.rand(768).astype('float32')
            embeddings.append(embedding)
        return np.array(embeddings)
    
    router = SemanticRouter(
        name="test_router",
        embedding_method=mock_embedding_method,
        distance_threshold=0.5  # 增大阈值以更容易匹配
    )
    
    # 清除之前的路由
    print("Clearing routes...")
    router.clear_routes()
    
    # 添加路由规则
    print("Adding routes...")
    success1 = router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    print(f"Greeting route added: {success1}")
    
    success2 = router.add_route(
        questions=["如何退货", "退货流程是什么"],
        target="refund"
    )
    print(f"Refund route added: {success2}")

    # 测试路由功能
    print("\nTesting routing...")
    result = router.route("Hi, good morning")
    print(f"Route result for 'Hi, good morning': {result}")
    
    result = router.route("退货流程是什么")
    print(f"Route result for '退货流程是什么': {result}")
    
    # 测试精确匹配
    result = router.route("如何退货")
    print(f"Route result for '如何退货': {result}")
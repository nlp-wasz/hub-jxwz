

import os
import json
import hashlib
import numpy as np
import redis
import faiss
from dataclasses import dataclass
from typing import Optional, List, Union, Callable, Any, Dict


@dataclass
class Route:
    """路由规则数据结构"""
    name: str                    # 路由名称
    references: List[str]        # 参考示例文本
    distance_threshold: float    # 距离阈值
    metadata: Dict[str, Any] = None  # 附加元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def mock_embedding_method(texts: Union[str, List[str]], dim: int = 768) -> np.ndarray:
    """
    伪embedding方法，用于演示语义路由逻辑

    Args:
        texts: 输入文本或文本列表
        dim: 向量维度

    Returns:
        numpy数组，形状为 (batch_size, dim)
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []

    for text in texts:
        # 基于文本长度和字符生成简单向量
        # 相似文本会产生相似向量
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # 使用哈希值生成向量的基础模式
        base_vector = np.zeros(dim)
        for i, char in enumerate(text_hash[:dim//4]):
            # 将字符转换为数值
            char_val = ord(char) % 256
            base_vector[i*4:(i*4+4)] = char_val / 256.0

        # 添加文本长度的影响
        length_factor = min(len(text) / 100.0, 1.0)  # 归一化长度影响
        base_vector[:10] *= (1 + length_factor * 0.1)

        # 添加一些随机性但保持一致性
        np.random.seed(hash(text) % 1000)
        noise = np.random.normal(0, 0.01, dim)
        base_vector += noise

        # 归一化
        base_vector = base_vector / np.linalg.norm(base_vector)

        embeddings.append(base_vector)

    return np.array(embeddings, dtype=np.float32)


class SemanticRouter:
    """
    语义路由器核心类

    基于向量相似性进行意图识别和路由决策
    与现有SemanticCache保持一致的接口风格
    """

    def __init__(
        self,
        name: str,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold: float = 0.3,
        cache_ttl: int = 3600
    ):
        """
        初始化语义路由器

        Args:
            name: 路由器名称，用于区分不同的路由器实例
            embedding_method: 向量嵌入方法，如果为None则使用默认的伪embedding
            ttl: 默认TTL过期时间（秒）
            redis_url: Redis服务器地址
            redis_port: Redis端口
            redis_password: Redis密码
            distance_threshold: 默认距离阈值
            cache_ttl: 路由结果缓存TTL（秒）
        """
        self.name = name
        self.default_distance_threshold = distance_threshold
        self.cache_ttl = cache_ttl

        # 设置embedding方法
        if embedding_method is None:
            self.embedding_method = mock_embedding_method
        else:
            self.embedding_method = embedding_method

        try:
            # Redis连接，与现有组件保持一致
            self.redis = redis.Redis(
                host=redis_url,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
            # 测试连接
            self.redis.ping()
        except Exception as e:
            print(f"Warning: Redis连接失败 {e}")
            self.redis = None

        # FAISS索引文件路径
        self.index_file = f"{name}_semantic_router.index"

        # 加载或初始化FAISS索引
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                print(f"已加载路由索引，包含 {self.index.ntotal} 个向量")
            else:
                self.index = None
                print("创建新的路由索引")
        except Exception as e:
            print(f"Warning: 索引加载失败 {e}")
            self.index = None

        # 存储路由信息的内存缓存
        self._routes_cache = {}
        self._load_routes_from_redis()

    def _load_routes_from_redis(self):
        """从Redis加载路由信息到内存缓存"""
        if not self.redis:
            return

        try:
            routes_data = self.redis.hgetall(f"router:{self.name}:routes")
            for route_name, route_json in routes_data.items():
                route_dict = json.loads(route_json)
                self._routes_cache[route_name] = route_dict
        except Exception as e:
            print(f"Warning: 加载路由信息失败 {e}")

    def add_route(self, route: Route) -> bool:
        """
        添加单个路由规则

        Args:
            route: 路由规则对象

        Returns:
            bool: 添加是否成功
        """
        try:
            print(f"添加路由规则: {route.name}")

            # 1. 向量化参考文本
            embeddings = self.embedding_method(route.references)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # 2. 初始化或更新FAISS索引
            if self.index is None:
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)

            # 3. 记录添加前的向量数量
            start_idx = self.index.ntotal

            # 4. 添加向量到索引
            self.index.add(embeddings)

            # 5. 保存索引到文件
            faiss.write_index(self.index, self.index_file)

            # 6. 存储路由元数据到Redis
            route_data = {
                'name': route.name,
                'references': route.references,
                'metadata': route.metadata,
                'distance_threshold': route.distance_threshold,
                'start_idx': start_idx,
                'end_idx': self.index.ntotal - 1,
                'vector_count': len(route.references)
            }

            if self.redis:
                with self.redis.pipeline() as pipe:
                    pipe.hset(f"router:{self.name}:routes", route.name, json.dumps(route_data))
                    pipe.lpush(f"router:{self.name}:route_list", route.name)
                    pipe.execute()

            # 7. 更新内存缓存
            self._routes_cache[route.name] = route_data

            print(f"路由 {route.name} 添加成功，包含 {len(route.references)} 个参考文本")
            return True

        except Exception as e:
            print(f"添加路由失败: {e}")
            return False

    def route(self, query: str) -> Optional[Dict[str, Any]]:
        """
        对查询进行语义路由

        Args:
            query: 用户查询文本

        Returns:
            路由结果字典，包含路由名称、置信度等信息，如无匹配则返回None
        """
        try:
            # 1. 检查路由缓存
            if self.redis:
                cache_key = self._get_query_hash(query)
                cached_result = self.redis.get(f"router:{self.name}:cache:{cache_key}")
                if cached_result:
                    print(f"缓存命中: {query}")
                    return json.loads(cached_result)

            # 2. 检查索引是否存在
            if self.index is None or self.index.ntotal == 0:
                print("警告: 路由索引为空，无法进行路由")
                return None

            # 3. 向量化查询
            query_embedding = self.embedding_method([query])
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # 4. 向量相似性检索
            distances, indices = self.index.search(query_embedding, k=min(100, self.index.ntotal))

            # 5. 找到最佳匹配路由
            best_match = self._find_best_route_match(distances[0], indices[0])

            # 6. 缓存结果
            if best_match and self.redis:
                cache_key = self._get_query_hash(query)
                self.redis.setex(
                    f"router:{self.name}:cache:{cache_key}",
                    self.cache_ttl,
                    json.dumps(best_match)
                )

            if best_match:
                print(f"路由匹配成功: {query} -> {best_match['route_name']} "
                      f"(置信度: {best_match['confidence']:.2f})")
            else:
                print(f"未找到匹配的路由: {query}")

            return best_match

        except Exception as e:
            print(f"路由处理错误: {e}")
            return None

    def _find_best_route_match(self, distances: np.ndarray, indices: np.ndarray) -> Optional[Dict[str, Any]]:
        """找到最佳路由匹配"""
        best_match = None
        min_distance = float('inf')

        # 获取所有路由信息
        all_routes = self._routes_cache.copy()

        for distance, idx in zip(distances, indices):
            # 跳过超出默认阈值的结果
            if distance > self.default_distance_threshold:
                break

            # 找到对应的路由
            route_name = self._find_route_by_index(idx, all_routes)
            if not route_name:
                continue

            route = all_routes[route_name]
            route_threshold = route.get('distance_threshold', self.default_distance_threshold)

            # 检查是否满足路由特定的阈值
            if distance < route_threshold and distance < min_distance:
                confidence = max(0, 1 - distance / route_threshold)
                best_match = {
                    'route_name': route['name'],
                    'metadata': route['metadata'],
                    'distance': float(distance),
                    'confidence': confidence,
                    'matched_reference': self._get_reference_by_index(idx, route)
                }
                min_distance = distance

        return best_match

    def _find_route_by_index(self, idx: int, all_routes: Dict[str, Dict]) -> Optional[str]:
        """根据向量索引找到对应路由"""
        for route_name, route_data in all_routes.items():
            if route_data['start_idx'] <= idx <= route_data['end_idx']:
                return route_name
        return None

    def _get_reference_by_index(self, idx: int, route: Dict) -> str:
        """根据索引获取参考文本"""
        ref_idx = idx - route['start_idx']
        if 0 <= ref_idx < len(route['references']):
            return route['references'][ref_idx]
        return ""

    def _get_query_hash(self, query: str) -> str:
        """生成查询哈希值用于缓存键"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def list_routes(self) -> List[Dict[str, Any]]:
        """
        列出所有路由规则

        Returns:
            路由规则列表
        """
        routes = []
        for route_name, route_data in self._routes_cache.items():
            routes.append({
                'name': route_data['name'],
                'reference_count': len(route_data['references']),
                'distance_threshold': route_data['distance_threshold'],
                'metadata': route_data['metadata']
            })
        return routes

    def clear_cache(self) -> bool:
        """
        清空路由缓存

        Returns:
            清空是否成功
        """
        try:
            if self.redis:
                cache_keys = self.redis.keys(f"router:{self.name}:cache:*")
                if cache_keys:
                    self.redis.delete(*cache_keys)
                    print(f"已清空 {len(cache_keys)} 个缓存项")
            return True
        except Exception as e:
            print(f"清空缓存失败: {e}")
            return False

    def delete_route(self, route_name: str) -> bool:
        """
        删除路由规则

        Args:
            route_name: 要删除的路由名称

        Returns:
            删除是否成功
        """
        try:
            if route_name not in self._routes_cache:
                print(f"路由不存在: {route_name}")
                return False

            # 从Redis删除
            if self.redis:
                with self.redis.pipeline() as pipe:
                    pipe.hdel(f"router:{self.name}:routes", route_name)
                    pipe.lrem(f"router:{self.name}:route_list", 0, route_name)
                    pipe.execute()

            # 从内存缓存删除
            del self._routes_cache[route_name]

            # 重建索引（简化实现）
            print("重建路由索引...")
            self._rebuild_index()

            print(f"路由删除成功: {route_name}")
            return True

        except Exception as e:
            print(f"删除路由失败: {e}")
            return False

    def _rebuild_index(self):
        """重建FAISS索引"""
        try:
            if not self._routes_cache:
                self.index = None
                if os.path.exists(self.index_file):
                    os.unlink(self.index_file)
                return

            # 重新构建索引
            all_embeddings = []
            for route_data in self._routes_cache.values():
                embeddings = self.embedding_method(route_data['references'])
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                all_embeddings.append(embeddings)

            if all_embeddings:
                all_embeddings = np.vstack(all_embeddings)
                dim = all_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(all_embeddings)
                faiss.write_index(self.index, self.index_file)
                print(f"索引重建完成，包含 {self.index.ntotal} 个向量")

        except Exception as e:
            print(f"重建索引失败: {e}")
            self.index = None


# 使用示例和测试代码
if __name__ == "__main__":
    def demo_semantic_router():
        """SemanticRouter使用演示"""
        print("=== SemanticRouter 演示 ===")

        # 1. 创建路由器
        router = SemanticRouter(
            name="demo_router",
            distance_threshold=0.4,
            cache_ttl=300  # 5分钟缓存
        )

        # 2. 定义路由规则
        routes = [
            Route(
                name="greeting",
                references=["你好", "您好", "hello", "hi", "早上好", "下午好", "晚上好"],
                distance_threshold=0.5,
                metadata={"type": "greeting", "response_type": "friendly"}
            ),
            Route(
                name="farewell",
                references=["再见", "bye", "goodbye", "拜拜", "下次见", "回头见"],
                distance_threshold=0.4,
                metadata={"type": "farewell", "response_type": "polite"}
            ),
            Route(
                name="weather",
                references=["天气", "weather", "下雨", "晴天", "阴天", "气温", "温度"],
                distance_threshold=0.4,
                metadata={"type": "weather", "response_type": "informative"}
            ),
            Route(
                name="help_request",
                references=["帮助", "help", "协助", "support", "怎么办", "如何"],
                distance_threshold=0.5,
                metadata={"type": "help", "response_type": "supportive"}
            ),
            Route(
                name="question",
                references=["什么", "如何", "怎么", "为什么", "where", "what", "how", "when"],
                distance_threshold=0.6,
                metadata={"type": "question", "response_type": "informative"}
            )
        ]

        # 3. 添加路由规则
        print("\n添加路由规则...")
        for route in routes:
            success = router.add_route(route)
            print(f"添加 {route.name}: {'成功' if success else '失败'}")

        # 4. 列出所有路由
        print(f"\n当前路由列表:")
        for route_info in router.list_routes():
            print(f"  - {route_info['name']}: {route_info['reference_count']} 个参考文本")

        # 5. 测试路由匹配
        test_queries = [
            "你好，最近怎么样？",
            "今天天气如何？",
            "请问你能帮我什么忙？",
            "再见，下次再聊！",
            "什么是人工智能？",
            "hello there",
            "会下雨吗？"
        ]

        print(f"\n开始路由测试...")
        print("-" * 60)

        for query in test_queries:
            result = router.route(query)
            if result:
                print(f"查询: {query}")
                print(f"路由: {result['route_name']}")
                print(f"置信度: {result['confidence']:.2f}")
                print(f"距离: {result['distance']:.4f}")
                print(f"匹配参考: {result['matched_reference']}")
                print(f"元数据: {result['metadata']}")
            else:
                print(f"查询: {query} -> 未找到匹配路由")
            print("-" * 60)

        # 6. 测试缓存效果（第二次查询相同的文本）
        print(f"\n测试缓存效果...")
        cached_query = "你好，最近怎么样？"
        result = router.route(cached_query)
        print(f"缓存查询结果: {result['route_name'] if result else 'None'}")

        # 7. 展示路由管理功能
        print(f"\n路由管理演示...")

        # 删除一个路由
        print("删除 'weather' 路由...")
        router.delete_route("weather")

        # 再次测试天气相关查询
        weather_query = "今天天气怎么样？"
        result = router.route(weather_query)
        print(f"删除后查询 '{weather_query}': {result['route_name'] if result else 'None'}")

        # 清空缓存
        print("\n清空缓存...")
        router.clear_cache()

        print("\n演示完成！")

    # 运行演示
    demo_semantic_router()
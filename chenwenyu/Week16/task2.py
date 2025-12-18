import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold: float = 0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True  # ✅ 添加这个参数，自动解码字符串
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        
        # 检查并创建索引目录
        self.index_file = f"{self.name}.index"
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"✅ 从 {self.index_file} 加载现有索引")
        else:
            self.index = None
            print(f"⚠️ 未找到现有索引，将创建新索引")

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        """
        存储提示和对应的响应
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]
        
        # 生成向量
        embeddings = self.embedding_method(prompt)
        
        # 初始化或更新 Faiss 索引
        if self.index is None:
            # 获取向量维度
            if isinstance(embeddings, list):
                dim = len(embeddings[0])
            elif isinstance(embeddings, np.ndarray):
                dim = embeddings.shape[1]
            else:
                raise ValueError(f"不支持的向量类型: {type(embeddings)}")
            
            self.index = faiss.IndexFlatL2(dim)
            print(f"✅ 创建新的 Faiss 索引，维度: {dim}")
        
        # 转换为 numpy array 并添加到索引
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings).astype('float32')
        
        self.index.add(embeddings)
        
        # 保存索引到文件
        faiss.write_index(self.index, self.index_file)
        print(f"✅ Faiss 索引已保存到 {self.index_file}，当前包含 {self.index.ntotal} 个向量")
        
        # 存储到 Redis
        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    # 存储键值对
                    cache_key = f"{self.name}:key:{q}"
                    pipe.setex(cache_key, self.ttl, a)
                    
                    # 存储提示列表
                    list_key = f"{self.name}:list"
                    pipe.lpush(list_key, q)
                
                results = pipe.execute()
                print(f"✅ 存储成功: {len(prompt)} 个提示")
                return results
        except Exception as e:
            print(f"❌ 存储失败: {e}")
            import traceback
            traceback.print_exc()
            return -1

    def check(self, prompt: str) -> Optional[str]:
        """
        检查缓存，返回最相似的响应
        """
        if self.index is None or self.index.ntotal == 0:
            return None
        
        # 生成查询向量
        embedding = self.embedding_method(prompt)
        
        # 转换为 numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding).astype('float32')
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # 搜索最相似的向量
        k = min(10, self.index.ntotal)  # 最多搜索10个
        distances, indices = self.index.search(embedding, k)
        
        # 检查是否有足够相似的结果
        if distances[0][0] > self.distance_threshold:
            return None
        
        # 获取所有满足阈值的结果
        valid_indices = [i for i, d in enumerate(distances[0]) 
                        if d <= self.distance_threshold]
        
        if not valid_indices:
            return None
        
        # 获取对应的提示
        list_key = f"{self.name}:list"
        all_prompts = self.redis.lrange(list_key, 0, -1)
        
        if not all_prompts or len(all_prompts) <= max(valid_indices):
            return None
        
        # 获取最相似的提示
        best_prompt = all_prompts[valid_indices[0]]
        
        # 获取对应的响应
        cache_key = f"{self.name}:key:{best_prompt}"
        response = self.redis.get(cache_key)
        
        if response:
            print(f"✅ 缓存命中: 相似度 {1 - distances[0][valid_indices[0]]:.4f}")
        
        return response

    def clear_cache(self):
        """
        清除所有缓存数据
        """
        print("🧹 正在清除缓存...")
        
        try:
            # 1. 获取所有提示
            list_key = f"{self.name}:list"
            prompts = self.redis.lrange(list_key, 0, -1)
            
            if prompts:
                # 2. 删除所有键值对
                pipe = self.redis.pipeline()
                for prompt in prompts:
                    cache_key = f"{self.name}:key:{prompt}"
                    pipe.delete(cache_key)
                
                # 3. 删除列表本身
                pipe.delete(list_key)
                pipe.execute()
                print(f"✅ 已删除 Redis 缓存: {len(prompts)} 个键")
            else:
                print("ℹ️ Redis 中没有缓存数据")
            
            # 4. 删除 Faiss 索引文件
            if os.path.exists(self.index_file):
                os.unlink(self.index_file)
                print(f"✅ 已删除 Faiss 索引文件: {self.index_file}")
            
            # 5. 重置索引
            self.index = None
            
            print("✅ 缓存清除完成")
            
        except Exception as e:
            print(f"❌ 清除缓存失败: {e}")
            import traceback
            traceback.print_exc()

    def info(self):
        """
        获取缓存信息
        """
        list_key = f"{self.name}:list"
        count = self.redis.llen(list_key)
        
        info = {
            "name": self.name,
            "redis_keys": count,
            "faiss_vectors": self.index.ntotal if self.index else 0,
            "distance_threshold": self.distance_threshold,
            "ttl": self.ttl
        }
        
        return info

    def get_all_keys(self):
        """
        获取所有缓存键（用于调试）
        """
        list_key = f"{self.name}:list"
        prompts = self.redis.lrange(list_key, 0, -1)
        keys = [f"{self.name}:key:{p}" for p in prompts]
        return keys


# 使用示例
if __name__ == "__main__":
    # 简单的向量生成函数（模拟）
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        
        # 生成随机向量（模拟真实 embedding）
        embeddings = []
        for t in text:
            # 使用文本哈希创建伪随机但确定的向量
            import hashlib
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)
            vec = np.random.randn(768).astype('float32')
            vec = vec / np.linalg.norm(vec)  # 归一化
            embeddings.append(vec)
        
        return np.array(embeddings)

    # 创建缓存
    cache = SemanticCache(
        name="test_cache",
        embedding_method=get_embedding,
        ttl=3600,  # 1小时
        redis_url="localhost",
        distance_threshold=0.3  # 相似度阈值
    )
    
    # 清除旧缓存
    cache.clear_cache()
    
    # 存储示例数据
    print("\n📝 存储示例数据...")
    cache.store(
        prompt="什么是机器学习？",
        response="机器学习是人工智能的一个分支，让计算机从数据中学习模式"
    )
    
    cache.store(
        prompt="如何学习Python编程？",
        response="学习Python可以从基础语法开始，然后学习数据结构、算法等"
    )
    
    cache.store(
        prompt="机器学习的基本原理是什么？",  # 与第一个问题相似
        response="机器学习通过算法让计算机从数据中发现规律和模式"
    )
    
    # 获取缓存信息
    info = cache.info()
    print(f"\n📊 缓存信息: {info}")
    
    # 测试缓存查询
    print("\n🔍 测试缓存查询...")
    
    # 测试1：精确匹配
    print("测试1 - 精确查询:")
    result = cache.check("什么是机器学习？")
    print(f"  查询: '什么是机器学习？'")
    print(f"  结果: {result}")
    
    # 测试2：相似查询
    print("\n测试2 - 相似查询:")
    result = cache.check("机器学习是什么？")  # 相似的问法
    print(f"  查询: '机器学习是什么？'")
    print(f"  结果: {result}")
    
    # 测试3：不同查询
    print("\n测试3 - 不同查询:")
    result = cache.check("如何做红烧肉？")  # 完全不同的主题
    print(f"  查询: '如何做红烧肉？'")
    print(f"  结果: {result}")
    
    # 查看所有键
    print(f"\n🗝️ 所有缓存键: {cache.get_all_keys()}")
    
    # 清理缓存
    print("\n🧹 清理缓存...")
    cache.clear_cache()
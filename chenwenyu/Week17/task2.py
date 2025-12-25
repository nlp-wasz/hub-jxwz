import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

class GPT4Rec:
    def __init__(self, model_name='/root/autodl-tmp/models/AI-modelScope/gpt2'):
        """
        初始化GPT4Rec推荐系统
        
        参数:
        model_name: 使用的GPT模型名称
        """
        # 加载GPT模型用于查询生成
        print("加载GPT模型...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # 设置pad_token为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 初始化数据存储
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_vectorizer = None
        self.movie_vectors = None
        
    def load_data(self, ratings_path: str, movies_path: str):
        """
        加载评分和电影数据
        
        参数:
        ratings_path: 评分文件路径
        movies_path: 电影文件路径
        """
        print("加载数据...")
        
        # 加载评分数据
        self.ratings_df = pd.read_csv(ratings_path, sep="::", header=None, engine='python')
        self.ratings_df.columns = ["user_id", "movie_id", "rating", "timestamp"]
        
        # 加载电影数据
        self.movies_df = pd.read_csv(movies_path, sep="::", header=None, engine='python', encoding="latin")
        self.movies_df.columns = ["movie_id", "title", "genres"]
        
        print(f"加载完成: {len(self.ratings_df)} 条评分, {len(self.movies_df)} 部电影")
        
    def prepare_retrieval_system(self):
        """
        准备基于TF-IDF的检索系统
        """
        print("准备检索系统...")
        
        # 创建电影文本描述（标题 + 类型）
        self.movies_df['text_description'] = self.movies_df['title'] + ' ' + self.movies_df['genres']
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 训练TF-IDF模型并转换电影描述
        self.movie_vectors = self.tfidf_vectorizer.fit_transform(
            self.movies_df['text_description']
        )
        
        print(f"检索系统准备完成，词汇表大小: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def get_user_history(self, user_id: int, top_n: int = 5) -> List[str]:
        """
        获取用户的历史观看记录
        
        参数:
        user_id: 用户ID
        top_n: 返回的历史记录数量
        
        返回:
        用户观看过的电影标题列表
        """
        # 获取用户的评分记录
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        # 按评分和时间排序（假设评分高且近期观看的更重要）
        user_ratings = user_ratings.sort_values(['rating', 'timestamp'], ascending=[False, False])
        
        # 获取前top_n部电影
        top_movies = user_ratings.head(top_n)
        
        # 获取电影标题
        history_movies = []
        for _, row in top_movies.iterrows():
            movie_info = self.movies_df[self.movies_df['movie_id'] == row['movie_id']]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0]['genres']
                history_movies.append(f"{title} ({genres})")
        
        return history_movies
    
    def generate_search_query(self, user_history: List[str]) -> str:
        """
        使用GPT生成搜索查询
        
        参数:
        user_history: 用户历史观看的电影列表
        
        返回:
        生成的搜索查询
        """
        # 构建提示词
        history_text = "\n".join([f"- {movie}" for movie in user_history])
        
        prompt = f"""你是一个电影推荐专家。请根据用户历史观看的电影，生成一个简短的搜索查询来描述用户未来的兴趣。

用户历史观看的电影：
{history_text}

基于上述电影，用户未来可能对什么样的电影感兴趣？请生成一个简短的搜索查询（不超过15个单词）来描述用户可能喜欢的电影类型。
搜索查询应该：
1. 使用中文
2. 包含电影类型关键词
3. 包含主题关键词
4. 清晰可读

搜索查询：
"""
        
        # 使用GPT生成查询
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=768,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取查询部分（提示词之后的内容）
        query = generated_text[len(prompt):].strip()
        
        # 清理查询文本
        query = query.split('\n')[0]  # 取第一行
        query = query.replace('"', '').replace("'", '')
        
        return query
    
    def retrieve_movies(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        基于查询检索相关电影
        
        参数:
        query: 搜索查询
        top_k: 返回的电影数量
        
        返回:
        相关电影的DataFrame
        """
        # 将查询转换为TF-IDF向量
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.movie_vectors).flatten()
        
        # 获取最相似电影的索引
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 获取电影信息
        recommendations = self.movies_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['movie_id', 'title', 'genres', 'similarity_score']]
    
    def recommend_for_user(self, user_id: int, top_n_history: int = 5, top_k_recommend: int = 10) -> Dict:
        """
        为用户生成推荐
        
        参数:
        user_id: 用户ID
        top_n_history: 用于生成查询的历史记录数量
        top_k_recommend: 推荐数量
        
        返回:
        包含推荐结果的字典
        """
        print(f"\n为用户 {user_id} 生成推荐...")
        
        # 步骤1: 获取用户历史
        print("步骤1: 获取用户历史观看记录...")
        user_history = self.get_user_history(user_id, top_n_history)
        
        if not user_history:
            print(f"用户 {user_id} 没有历史记录")
            return {"user_id": user_id, "history": [], "query": "", "recommendations": []}
        
        print(f"用户历史: {user_history}")
        
        # 步骤2: 生成搜索查询
        print("步骤2: 生成搜索查询...")
        search_query = self.generate_search_query(user_history)
        print(f"生成的查询: {search_query}")
        
        # 步骤3: 检索相关电影
        print("步骤3: 检索相关电影...")
        recommendations = self.retrieve_movies(search_query, top_k_recommend)
        
        # 过滤掉用户已经看过的电影
        watched_movies = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].tolist()
        recommendations = recommendations[~recommendations['movie_id'].isin(watched_movies)]
        
        print(f"推荐完成！找到 {len(recommendations)} 部相关电影")
        
        return {
            "user_id": user_id,
            "history": user_history,
            "query": search_query,
            "recommendations": recommendations.to_dict('records')
        }
    
    def evaluate_recommendations(self, user_id: int, test_size: float = 0.2):
        """
        简单评估推荐效果（留一法评估）
        
        参数:
        user_id: 用户ID
        test_size: 测试集比例
        """
        # 获取用户的所有评分
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if len(user_ratings) < 10:
            print(f"用户 {user_id} 的评分记录不足，无法评估")
            return
        
        # 按时间排序
        user_ratings = user_ratings.sort_values('timestamp')
        
        # 分割训练集和测试集
        split_idx = int(len(user_ratings) * (1 - test_size))
        train_ratings = user_ratings.iloc[:split_idx]
        test_ratings = user_ratings.iloc[split_idx:]
        
        # 临时修改用户的评分数据
        original_ratings = self.ratings_df.copy()
        self.ratings_df = self.ratings_df[self.ratings_df['user_id'] != user_id]
        self.ratings_df = pd.concat([self.ratings_df, train_ratings])
        
        # 基于训练集生成推荐
        user_history = self.get_user_history(user_id, top_n=10)
        search_query = self.generate_search_query(user_history)
        recommendations = self.retrieve_movies(search_query, top_k=20)
        
        # 恢复原始数据
        self.ratings_df = original_ratings
        
        # 计算命中率
        test_movies = test_ratings['movie_id'].tolist()
        recommended_movies = recommendations['movie_id'].tolist()
        
        hits = set(test_movies) & set(recommended_movies)
        hit_rate = len(hits) / len(test_movies) if test_movies else 0
        
        print(f"评估结果:")
        print(f"- 测试集大小: {len(test_ratings)} 部电影")
        print(f"- 命中电影: {len(hits)} 部")
        print(f"- 命中率: {hit_rate:.2%}")
     
        return hit_rate

def main():
    """主函数"""
    # 初始化GPT4Rec
    recommender = GPT4Rec(model_name='/root/autodl-tmp/models/AI-ModelScope/gpt2')
    
    # 加载数据（请确保路径正确）
    try:
        recommender.load_data(
            ratings_path="./M_ML-100K/ratings.dat",
            movies_path="./M_ML-100K/movies.dat"
        )
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在：")
        print("1. ./M_ML-100K/ratings.dat")
        print("2. ./M_ML-100K/movies.dat")
        return
    
    # 准备检索系统
    recommender.prepare_retrieval_system()
    
    # 示例：为用户生成推荐
    # 选择一个有足够历史记录的用户
    user_counts = recommender.ratings_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 20].index.tolist()
    
    if active_users:
        # 为前5个活跃用户生成推荐
        for i, user_id in enumerate(active_users[:5]):
            print(f"\n{'='*60}")
            print(f"处理用户 {i+1}: ID={user_id}")
            
            # 生成推荐
            result = recommender.recommend_for_user(
                user_id=user_id,
                top_n_history=5,
                top_k_recommend=10
            )
            
            # 显示结果
            print(f"\n最终推荐结果:")
            for j, movie in enumerate(result['recommendations'][:5], 1):
                print(f"{j}. {movie['title']} (相似度: {movie['similarity_score']:.3f})")
            
            # 评估推荐效果
            if i < 3:  # 只评估前3个用户
                print(f"\n评估推荐效果:")
                recommender.evaluate_recommendations(user_id)
    
    # 交互式推荐
    print(f"\n{'='*60}")
    print("交互式推荐测试")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n请输入用户ID进行推荐（输入 'q' 退出）: ")
            
            if user_input.lower() == 'q':
                print("再见！")
                break
            
            user_id = int(user_input)
            
            # 检查用户是否存在
            if user_id not in recommender.ratings_df['user_id'].values:
                print(f"用户 {user_id} 不存在，请重试")
                continue
            
            # 生成推荐
            result = recommender.recommend_for_user(user_id)
            
            if result['recommendations']:
                print(f"\n为用户 {user_id} 的推荐:")
                print(f"生成的搜索查询: {result['query']}")
                print(f"\n推荐电影:")
                for i, movie in enumerate(result['recommendations'][:10], 1):
                    print(f"{i:2d}. {movie['title']} ({movie['genres']}) - 相似度: {movie['similarity_score']:.3f}")
            else:
                print(f"无法为用户 {user_id} 生成推荐")
                
        except ValueError:
            print("请输入有效的数字ID")
        except KeyboardInterrupt:
            print("\n程序已终止")
            break


if __name__ == "__main__":
    main()

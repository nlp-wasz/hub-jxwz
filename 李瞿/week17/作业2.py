'''
GPT4Rec模型：基于生成式语言模型的个性化推荐框架，借助GPT生成未来的查询条件，使用搜索检索到相关的物品。
    - 步骤1：（生成查询条件）根据用户历史交互物品的文本信息如商品标题，生成能够代表用户未来兴趣的，可读的搜索查询。
    - 步骤2：（物品的检索）：从整个物品库中检索出最相关的物品作为推荐候选
'''

import pandas as pd
import numpy as np
import dashscope
import os
import re
import time

# 设置DashScope API密钥 (请替换为您的实际API密钥)
# 您可以从环境变量中设置，或者直接在这里设置
dashscope.api_key = "sk-facd0ca4f5ae4fada1706bf3938b69d9"

# 读取MovieLens 100K数据
rating = pd.read_csv(r"D:\code\ba_dou\Week17\03_推荐系统\M_ML-100K\ratings.dat", sep="::", header=None, engine='python', encoding='latin1')
rating.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv(r"D:\code\ba_dou\Week17\03_推荐系统\M_ML-100K\movies.dat", sep="::", header=None, engine='python', encoding='latin1')
movies.columns = ['movie_id', 'movie_title', 'movie_tag']

class GPT4Rec:
    def __init__(self, rating_df, movies_df):
        self.rating_df = rating_df
        self.movies_df = movies_df
        self.user_profiles = {}  # 存储用户画像
        
    def _get_user_history(self, user_id, min_rating=4):
        """
        获取用户历史观看的电影
        """
        # 获取用户评分过的电影 (只考虑评分>=min_rating的电影)
        user_movies = self.rating_df[(self.rating_df['user_id'] == user_id) & (self.rating_df['rating'] >= min_rating)]
        # 合并电影信息
        user_history = pd.merge(user_movies, self.movies_df, on='movie_id')
        return user_history.sort_values('rating', ascending=False)
    
    def _generate_user_profile(self, user_id):
        """
        生成用户画像：基于用户历史行为构建用户兴趣表示
        """
        user_history = self._get_user_history(user_id)
        
        if len(user_history) == 0:
            return None
            
        # 提取用户最喜欢的类型
        all_tags = " ".join(user_history['movie_tag'].tolist())
        genre_count = {}
        for tag in all_tags.split('|'):
            if tag:
                genre_count[tag] = genre_count.get(tag, 0) + 1
        
        # 按频率排序
        sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        favorite_genres = [genre for genre, count in sorted_genres[:5]]
        
        # 获取高评分电影
        top_rated = user_history.head(10)
        
        profile = {
            'user_id': user_id,
            'favorite_genres': favorite_genres,
            'top_rated_movies': top_rated[['movie_title', 'movie_tag', 'rating']].to_dict('records'),
            'history_count': len(user_history)
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def _generate_qwen_prompt(self, user_profile):
        """
        根据用户画像生成Qwen提示词
        """
        if not user_profile:
            return "为一位新用户推荐10部经典高分电影。"
            
        genres_str = ", ".join(user_profile['favorite_genres']) if user_profile['favorite_genres'] else "多种类型"
        
        prompt = f"""
你是一位专业的电影推荐专家。基于以下用户观影历史和偏好，为该用户推荐10部他们可能喜欢但尚未观看的电影。

用户偏好分析：
1. 最喜欢的电影类型：{genres_str}
2. 历史观影数量：{user_profile['history_count']}部

用户高分评价电影（评分从高到低）：
"""
        
        for i, movie in enumerate(user_profile['top_rated_movies'][:5], 1):
            prompt += f"{i}. 《{movie['movie_title']}》 评分：{movie['rating']}\n"
        
        prompt += """
请基于以上信息，推荐10部符合该用户品味但其尚未观看的电影。按照以下格式输出，每行一部电影：

1. 电影名称1
2. 电影名称2
...
10. 电影名称10

只输出电影名称列表，不需要其他解释文字。
"""
        return prompt
    
    def _call_qwen_api(self, prompt, model="qwen-max"):
        """
        调用Qwen API生成推荐
        """
        try:
            response = dashscope.Generation.call(
                model=model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                print(f"调用Qwen API失败: {response}")
                return None
        except Exception as e:
            print(f"调用Qwen API时出错: {e}")
            return None
    
    def _parse_qwen_response(self, response_text):
        """
        解析Qwen响应结果
        """
        if not response_text:
            return []
            
        # 按行分割并清理
        lines = response_text.strip().split('\n')
        movie_names = []
        
        for line in lines:
            # 移除编号和前后空格
            line = re.sub(r'^\d+\.\s*', '', line.strip())
            if line:
                movie_names.append(line)
                
        return movie_names[:10]  # 最多返回10部电影
    
    def _find_movies_by_names(self, movie_names):
        """
        根据电影名称查找电影信息
        """
        recommendations = []
        for name in movie_names:
            # 精确匹配
            matched = self.movies_df[self.movies_df['movie_title'] == name]
            if not matched.empty:
                recommendations.append(matched.iloc[0])
            else:
                # 模糊匹配
                matched = self.movies_df[self.movies_df['movie_title'].str.contains(name, case=False, na=False)]
                if not matched.empty:
                    recommendations.append(matched.iloc[0])
                    
        return recommendations
    
    def _fallback_recommend(self, user_id, top_k=10):
        """
        回退推荐策略：基于协同过滤或热门电影
        """
        # 基于用户历史中的高评分电影类型推荐相似电影
        user_history = self._get_user_history(user_id)
        
        if len(user_history) > 0:
            # 获取用户喜欢的类型中最受欢迎的电影
            all_tags = " ".join(user_history['movie_tag'].tolist())
            favorite_genre = ""
            max_count = 0
            
            for tag in all_tags.split('|'):
                if tag:
                    tag_movies = self.movies_df[self.movies_df['movie_tag'].str.contains(tag, case=False, na=False)]
                    if len(tag_movies) > max_count:
                        max_count = len(tag_movies)
                        favorite_genre = tag
            
            # 推荐该类型的高分电影
            if favorite_genre:
                genre_movies = self.movies_df[
                    (self.movies_df['movie_tag'].str.contains(favorite_genre, case=False, na=False)) &
                    (~self.movies_df['movie_id'].isin(user_history['movie_id']))
                ]
                
                # 计算这些电影的平均评分
                genre_ratings = self.rating_df[self.rating_df['movie_id'].isin(genre_movies['movie_id'])]
                avg_ratings = genre_ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
                avg_ratings = avg_ratings[avg_ratings['count'] >= 10]  # 至少有10个评分
                
                # 合并信息并排序
                result = pd.merge(genre_movies, avg_ratings, on='movie_id', how='inner')
                result = result.sort_values(['mean', 'count'], ascending=False).head(top_k)
                
                return result.to_dict('records')
        
        # 如果还是无法推荐，则推荐全局热门电影
        popular_movies = self.rating_df.groupby('movie_id').agg({'rating': ['mean', 'count']})
        popular_movies.columns = ['avg_rating', 'rating_count']
        popular_movies = popular_movies[popular_movies['rating_count'] >= 50]
        popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=False)
        
        recommended = pd.merge(popular_movies.head(top_k).reset_index(), self.movies_df, on='movie_id')
        return recommended.to_dict('records')
    
    def recommend(self, user_id, top_k=10, use_qwen=True):
        """
        为用户生成推荐
        """
        # 生成用户画像
        user_profile = self._generate_user_profile(user_id)
        
        # 生成Qwen提示词
        prompt = self._generate_qwen_prompt(user_profile)
        
        print("发送给Qwen的提示词:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
        
        qwen_response = None
        if use_qwen:
            # 调用Qwen API
            qwen_response = self._call_qwen_api(prompt)
            
            if qwen_response:
                print("\nQwen生成的推荐结果:")
                print("=" * 50)
                print(qwen_response)
                print("=" * 50)
            else:
                print("\n调用Qwen API失败，使用回退推荐策略")
        
        recommendations = []
        if qwen_response:
            # 解析Qwen响应
            movie_names = self._parse_qwen_response(qwen_response)
            
            # 查找对应的电影信息
            recommendations = self._find_movies_by_names(movie_names)
        
        # 如果推荐数量不足或Qwen调用失败，使用回退策略补充
        if len(recommendations) < top_k:
            fallback_recs = self._fallback_recommend(user_id, top_k - len(recommendations))
            for rec in fallback_recs:
                # 转换为Series格式以保持一致性
                series_rec = pd.Series(rec)
                recommendations.append(series_rec)
                
        return recommendations[:top_k]

def main():
    # 检查是否设置了API密钥
    if not dashscope.api_key or dashscope.api_key == "sk-facd0ca4f5ae4fada1706bf3938b69d9":
        print("警告: 请替换代码中的占位符API密钥为您自己的阿里云API密钥")
        print("将使用回退推荐策略演示系统功能")
        use_qwen = False
    else:
        print("已检测到DashScope API密钥，将使用Qwen模型生成推荐")
        use_qwen = True
    
    # 初始化GPT4Rec系统
    gpt4rec = GPT4Rec(rating, movies)
    
    # 为用户1生成推荐
    print("\nGPT4Rec 个性化电影推荐系统 (基于Qwen)")
    print("=" * 50)
    
    user_id = 1
    recommendations = gpt4rec.recommend(user_id, use_qwen=use_qwen)
    
    print(f"\n为用户 {user_id} 推荐的电影:")
    print("-" * 30)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['movie_title']}")
    
    print("\n" + "="*50)
    
    # 为用户100生成推荐
    user_id = 100
    recommendations = gpt4rec.recommend(user_id, use_qwen=use_qwen)
    
    print(f"\n为用户 {user_id} 推荐的电影:")
    print("-" * 30)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['movie_title']}")

if __name__ == "__main__":
    main()
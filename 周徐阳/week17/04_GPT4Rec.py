# 04_GPT4Rec.py
import pandas as pd
import numpy as np
from collections import Counter
import re

# ===== 数据加载 =====
print("正在加载数据...")
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]

print(f"数据加载完成: {len(ratings)} 条评分记录, {len(movies)} 部电影\n")

# ===== Prompt模板 =====
PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""

# ===== 步骤1：生成查询条件（模拟GPT生成） =====
def generate_query_from_history(user_history_titles, all_movies):
    """
    根据用户历史观看电影生成推荐查询
    这里使用基于内容的方法模拟GPT生成过程
    """
    # 提取用户历史电影的类型标签
    user_genres = []
    for title in user_history_titles:
        movie_info = all_movies[all_movies['movie_title'] == title]
        if not movie_info.empty:
            genres = movie_info.iloc[0]['movie_tag'].split('|')
            user_genres.extend(genres)
    
    # 统计最常见的类型
    genre_counter = Counter(user_genres)
    top_genres = [genre for genre, count in genre_counter.most_common(3)]
    
    # 提取电影标题中的关键词（年份、系列等）
    years = []
    for title in user_history_titles:
        year_match = re.findall(r'\((\d{4})\)', title)
        if year_match:
            years.append(int(year_match[0]))
    
    # 生成查询描述（模拟GPT输出）
    query_info = {
        'preferred_genres': top_genres,
        'year_range': (min(years) if years else 1990, max(years) if years else 2000),
        'history_titles': user_history_titles
    }
    
    return query_info

# ===== 步骤2：物品检索 =====
def retrieve_movies(query_info, all_movies, user_watched_ids, top_k=10):
    """
    基于生成的查询条件，从电影库中检索相关电影
    """
    candidate_movies = all_movies[~all_movies['movie_id'].isin(user_watched_ids)].copy()
    
    # 计算相关性分数
    scores = []
    for idx, row in candidate_movies.iterrows():
        score = 0
        movie_genres = row['movie_tag'].split('|')
        
        # 类型匹配度
        genre_match = len(set(movie_genres) & set(query_info['preferred_genres']))
        score += genre_match * 10
        
        # 年份相关性
        year_match = re.findall(r'\((\d{4})\)', row['movie_title'])
        if year_match:
            movie_year = int(year_match[0])
            year_diff = abs(movie_year - query_info['year_range'][1])
            score += max(0, 10 - year_diff / 2)
        
        scores.append(score)
    
    candidate_movies['score'] = scores
    recommended = candidate_movies.nlargest(top_k, 'score')
    
    return recommended

# ===== 主推荐流程 =====
def recommend_for_user(user_id, ratings_df, movies_df, top_k=10, history_limit=20):
    """
    为指定用户生成推荐
    """
    print(f"{'='*60}")
    print(f"为用户 {user_id} 生成推荐")
    print(f"{'='*60}\n")
    
    # 获取用户历史观看记录（评分>=4的电影）
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('timestamp', ascending=False)
    user_high_ratings = user_ratings[user_ratings['rating'] >= 4].head(history_limit)
    
    if len(user_high_ratings) == 0:
        print("用户没有高分评价的电影，无法生成推荐")
        return None
    
    # 获取电影标题
    user_movie_ids = user_high_ratings['movie_id'].tolist()
    user_movies = movies_df[movies_df['movie_id'].isin(user_movie_ids)]
    user_history_titles = user_movies['movie_title'].tolist()
    
    print("【用户历史高分电影】")
    for i, title in enumerate(user_history_titles[:10], 1):
        print(f"  {i}. {title}")
    if len(user_history_titles) > 10:
        print(f"  ... 共 {len(user_history_titles)} 部电影")
    print()
    
    # 生成Prompt（实际场景中发送给GPT）
    history_text = "\n".join([f"- {title}" for title in user_history_titles])
    prompt = PROMPT_TEMPLATE.format(history_text)
    print("【生成的Prompt】")
    print(prompt[:300] + "...\n")
    
    # 步骤1: 生成查询条件
    print("【步骤1：生成查询条件】")
    query_info = generate_query_from_history(user_history_titles, movies_df)
    print(f"  偏好类型: {', '.join(query_info['preferred_genres'])}")
    print(f"  年份范围: {query_info['year_range'][0]} - {query_info['year_range'][1]}")
    print()
    
    # 步骤2: 检索推荐电影
    print("【步骤2：检索推荐候选】")
    all_watched_ids = user_ratings['movie_id'].tolist()
    recommended_movies = retrieve_movies(query_info, movies_df, all_watched_ids, top_k)
    
    print(f"\n推荐结果 (Top {top_k}):")
    print(f"{'-'*60}")
    for i, (idx, row) in enumerate(recommended_movies.iterrows(), 1):
        print(f"{i:2d}. {row['movie_title']:<45} | 类型: {row['movie_tag']:<30} | 分数: {row['score']:.1f}")
    
    return recommended_movies

# ===== 执行推荐 =====
if __name__ == "__main__":
    # 选择一个活跃用户进行演示
    user_activity = ratings.groupby('user_id').size()
    active_users = user_activity[user_activity >= 50].index.tolist()
    
    # 为几个用户生成推荐
    test_users = active_users[:3] if len(active_users) >= 3 else [1, 2, 3]
    
    for user_id in test_users:
        try:
            recommend_for_user(user_id, ratings, movies, top_k=10, history_limit=15)
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"用户 {user_id} 推荐失败: {e}\n")
    
    print("\n推荐完成！")
    
    # ===== 可选：与真实GPT API集成示例 =====
    """
    如果要使用真实的GPT API，可以替换generate_query_from_history函数：
    
    import openai
    
    def generate_query_with_gpt(user_history_titles):
        history_text = "\\n".join([f"- {title}" for title in user_history_titles])
        prompt = PROMPT_TEMPLATE.format(history_text)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        generated_text = response.choices[0].message.content
        # 解析生成的电影描述，用于检索
        return generated_text
    """

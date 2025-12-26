"""
GPT4Rec ：基于生成式语言模型的个性化推荐框架，借助 GPT2 生成未来的查询条件，使用搜索检索到相关的物品。
- 步骤1（生成查询条件）: 根据用户历史交互物品的文本信息（如商品标题），生成能够代表用户未来兴趣的、可读的“搜索查询”。
    Previously, the customer has bought: <标题1>. <标题2>... In the future, the customer wants to buy
- 步骤2（物品的检索）: 从整个物品库中检索出最相关的物品作为推荐候选
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
from openai import OpenAI

# 读取数据
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]

# 使用原有的PROMPT_TEMPLATE（不修改内容）
PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""

os.environ["OPENAI_API_KEY"] = "sk-2****0f44"

# 配置Qwen API
# QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# QWEN_API_KEY = "sk-2****0f44"
# QWEN_API_KEY = os.getenv("OPENAI_API_KEY")


class SimpleGPT4Rec:
    def __init__(self, movies_df):
        """初始化推荐系统"""
        self.movies_df = movies_df
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._build_search_index()

    def _build_search_index(self):
        """构建电影搜索索引"""
        movie_texts = self.movies_df['movie_title'] + " " + self.movies_df['movie_tag'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(movie_texts)

    def generate_queries_with_qwen(self, user_history):
        """步骤1：使用Qwen生成搜索查询（直接使用原有PROMPT_TEMPLATE）"""
        history_titles = "\n".join([f"- {title}" for title in user_history])
        prompt = PROMPT_TEMPLATE.format(history_titles)

        try:
            # headers = {
            #     "Authorization": f"Bearer {QWEN_API_KEY}",
            #     "Content-Type": "application/json"
            # }
            # payload = {
            #     "model": "qwen-turbo",
            #     "messages": [
            #         {"role": "user", "content": prompt}
            #     ],
            #     "temperature": 0.7,
            #     "max_tokens": 200
            # }
            # response = requests.post(QWEN_API_URL, headers=headers,
            #                          data=json.dumps(payload), timeout=30)
            # print("状态码:", response.status_code)  # 404
            # print("响应头:", response.headers)
            # print("原始响应内容（前500字符）:", response.text[:500])
            # response_data = response.json()
            # recommendations_text = response_data['choices'][0]['message']['content']

            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            response = client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "你是一个专业的电影推荐专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            # 解析返回的电影推荐，将其作为查询条件
            recommendations_text = response.choices[0].message.content
            queries = [q.strip() for q in recommendations_text.split('\n') if q.strip() and len(q.strip()) > 5]
            return queries[:5]  # 返回前5个作为查询

        except Exception as e:
            print(f"Qwen API调用失败: {e}")
            # 简化版本：直接返回历史电影标题作为查询
            return user_history[:3]

    def search_movies(self, query, top_k=5):
        """步骤2：基于查询搜索电影"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        similar_indices = similarities.argsort()[::-1][:top_k]
        recommended_movies = self.movies_df.iloc[similar_indices].copy()
        recommended_movies['similarity_score'] = similarities[similar_indices]
        recommended_movies['generated_query'] = query

        return recommended_movies

    def recommend_for_user(self, user_id, top_k=10):
        """为指定用户生成推荐"""
        # 获取用户历史记录
        user_ratings = ratings[ratings['user_id'] == user_id]
        if len(user_ratings) == 0:
            print(f"用户 {user_id} 无历史记录")
            return pd.DataFrame()

        user_movies = user_ratings.merge(movies, on='movie_id')
        user_history = user_movies['movie_title'].tolist()[:5]  # 取最近5部电影

        print(f"用户 {user_id} 的历史电影: {user_history}")

        # 生成查询
        queries = self.generate_queries_with_qwen(user_history)
        print(f"生成的查询: {queries}")

        # 为每个查询搜索电影
        all_recommendations = []
        for query in queries:
            results = self.search_movies(query, top_k=3)  # 每个查询取3部电影
            all_recommendations.append(results)

        if not all_recommendations:
            return pd.DataFrame()

        # 合并结果
        final_recommendations = pd.concat(all_recommendations, ignore_index=True)
        final_recommendations = final_recommendations.drop_duplicates('movie_id')
        final_recommendations = final_recommendations.sort_values('similarity_score', ascending=False)

        return final_recommendations.head(top_k)


# 使用示例
def main():
    # 初始化推荐系统
    recommender = SimpleGPT4Rec(movies)

    # 测试用户
    test_user_id = 1

    # 生成推荐
    recommendations = recommender.recommend_for_user(test_user_id, top_k=10)

    if len(recommendations) > 0:
        print(f"\n为用户 {test_user_id} 的推荐结果:")
        print("=" * 60)
        for idx, row in recommendations.iterrows():
            print(f"{idx + 1}. {row['movie_title']}")
            print(f"   相似度: {row['similarity_score']:.3f}")
            print(f"   生成查询: {row['generated_query']}")
            print(f"   标签: {row['movie_tag']}")
            print("-" * 40)
    else:
        print("未找到合适的推荐")


if __name__ == "__main__":
    main()
import pandas as pd
import os
from openai import OpenAI
from typing import List, Tuple, Optional
import re
from dataclasses import dataclass
from tqdm import tqdm


# ===================== 配置类（统一管理参数，提升可维护性）=====================
@dataclass
class GPT4RecConfig:
    """GPT4Rec 配置类，集中管理所有参数"""
    # 数据路径配置
    RATINGS_PATH: str = "../03_推荐系统/M_ML-100K/ratings.dat"
    MOVIES_PATH: str = "../03_推荐系统/M_ML-100K/movies.dat"
    # LLM 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk")
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-max"
    TEMPERATURE: float = 0.7  # 控制生成多样性
    MAX_TOKENS: int = 800
    # 推荐配置
    USER_HISTORY_TOP_N: int = 10  # 取用户评分最高的N部电影作为历史
    NUM_QUERIES: int = 5  # 生成N个多样化查询（GPT4Rec多查询策略）
    RECOMMENDATION_NUM: int = 10  # 最终推荐电影数量
    # 检索配置
    MATCH_THRESHOLD: float = 0.3  # 模糊匹配阈值（后续可扩展为BM25评分）


# 初始化配置
config = GPT4RecConfig()


# ===================== 数据加载工具（分离数据逻辑，便于复用）=====================
class MovieDataLoader:
    """电影数据加载器，负责数据读取、预处理和缓存"""

    def __init__(self, config: GPT4RecConfig):
        self.config = config
        self.ratings: Optional[pd.DataFrame] = None
        self.movies: Optional[pd.DataFrame] = None
        self._load_data()  # 初始化时加载数据

    def _load_data(self):
        """加载并预处理数据"""
        # 加载评分数据
        self.ratings = pd.read_csv(
            self.config.RATINGS_PATH,
            sep="::",
            header=None,
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        # 加载电影数据（增强编码处理和数据清洗）
        self.movies = pd.read_csv(
            self.config.MOVIES_PATH,
            sep="::",
            header=None,
            engine='python',
            encoding='latin-1',
            names=['movie_id', 'movie_title', 'movie_tag']
        )

        # 数据清洗：去除标题为空或过长的电影
        self.movies = self.movies[
            (self.movies['movie_title'].notna()) &
            (self.movies['movie_title'].str.len() <= 400)
            ].reset_index(drop=True)

        # 预处理：提取电影年份（优化检索维度）
        self.movies['movie_year'] = self.movies['movie_title'].str.extract(r'\((\d{4})\)')
        self.movies['clean_title'] = self.movies['movie_title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    def get_user_history(self, user_id: int) -> Tuple[pd.DataFrame, List[int]]:
        """获取用户历史观看记录（按评分排序）"""
        user_ratings = self.ratings[
            self.ratings['user_id'] == user_id
            ].sort_values('rating', ascending=False).head(self.config.USER_HISTORY_TOP_N)

        watched_movie_ids = user_ratings['movie_id'].tolist()
        watched_movies = self.movies[self.movies['movie_id'].isin(watched_movie_ids)]

        return watched_movies, watched_movie_ids

    def get_available_movies(self, excluded_ids: List[int]) -> pd.DataFrame:
        """获取排除已观看电影后的可用电影库"""
        return self.movies[~self.movies['movie_id'].isin(excluded_ids)].copy()


# ===================== LLM 查询生成器（实现GPT4Rec多查询策略）=====================
class LLMQueryGenerator:
    """LLM查询生成器，生成多样化的用户兴趣查询"""

    def __init__(self, config: GPT4RecConfig):
        self.config = config
        self.client = OpenAI(
            api_key=self.config.OPENAI_API_KEY,
            base_url=self.config.LLM_BASE_URL
        )
        # 优化后的提示词（更贴合GPT4Rec的兴趣建模思路）
        self.PROMPT_TEMPLATE = """
你是专业的电影推荐分析师，需要基于用户历史观看记录，生成{num_queries}个多样化的电影搜索查询。
每个查询需精准反映用户的一个兴趣维度（如类型偏好、主题偏好、风格偏好等），查询需具体、可检索。

用户历史观看的电影：
{watched_movies}

生成要求：
1. 共生成{num_queries}个查询，每个查询占一行
2. 查询需多样化，覆盖不同兴趣角度（避免重复类型）
3. 查询格式简洁，无需额外说明（例："科幻片 - 太空探险主题 - 视觉特效出色"）
4. 基于历史电影的类型、主题、风格生成，不脱离用户兴趣
"""

    def generate_queries(self, watched_movies: pd.DataFrame) -> List[str]:
        """生成多样化的用户兴趣查询"""
        # 构建结构化的历史观看文本
        watched_text = "\n".join([
            f"- {row['clean_title']} | 类型：{row['movie_tag']} | 年份：{row['movie_year'] if pd.notna(row['movie_year']) else '未知'}"
            for _, row in watched_movies.iterrows()
        ])

        # 填充提示词
        prompt = self.PROMPT_TEMPLATE.format(
            num_queries=self.config.NUM_QUERIES,
            watched_movies=watched_text
        )

        # 调用LLM生成查询
        try:
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是精准的兴趣查询生成器，生成的查询需可直接用于电影检索"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )

            # 解析生成的查询（过滤空行和无效内容）
            queries = [
                line.strip() for line in response.choices[0].message.content.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ]

            # 确保查询数量符合配置（不足时补全，过多时截断）
            if len(queries) < self.config.NUM_QUERIES:
                queries += [f"补充查询 - {watched_movies['movie_tag'].iloc[0].split('|')[0]}类型 - 高评分"
                            for _ in range(self.config.NUM_QUERIES - len(queries))]
            return queries[:self.config.NUM_QUERIES]

        except Exception as e:
            print(f"⚠️ LLM查询生成失败：{str(e)}")
            # 降级策略：基于历史类型生成默认查询
            default_tags = watched_movies['movie_tag'].str.split('|').explode().unique()[:self.config.NUM_QUERIES]
            return [f"默认查询 - {tag.strip()}类型 - 经典作品" for tag in default_tags]


# ===================== 电影检索器（模拟GPT4Rec的BM25检索逻辑）=====================
class MovieRetriever:
    """电影检索器，基于生成的查询匹配电影库"""

    def __init__(self, data_loader: MovieDataLoader):
        self.data_loader = data_loader
        self.movies = data_loader.movies

    def _calculate_match_score(self, movie: pd.Series, query: str) -> float:
        """计算电影与查询的匹配分数（模拟BM25的语义匹配逻辑）"""
        query_keywords = re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]{2,}', query.lower())  # 提取关键词（2字以上）
        if not query_keywords:
            return 0.0

        # 匹配维度：标题、类型、年份
        match_count = 0
        movie_text = f"{movie['clean_title'].lower()} {movie['movie_tag'].lower()} {str(movie['movie_year']).lower()}"

        for keyword in query_keywords:
            if keyword in movie_text:
                match_count += 1

        # 计算匹配率（关键词匹配数/总关键词数）
        return match_count / len(query_keywords)

    def retrieve_movies(self, query: str, excluded_ids: List[int], top_k: int = 3) -> pd.DataFrame:
        """基于单个查询检索电影"""
        available_movies = self.data_loader.get_available_movies(excluded_ids)
        if available_movies.empty:
            return pd.DataFrame()

        # 计算所有可用电影的匹配分数
        available_movies['match_score'] = available_movies.apply(
            lambda x: self._calculate_match_score(x, query), axis=1
        )

        # 筛选匹配分数高于阈值的电影，并按分数排序
        matched_movies = available_movies[
            available_movies['match_score'] >= config.MATCH_THRESHOLD
            ].sort_values('match_score', ascending=False).head(top_k)

        return matched_movies[['movie_id', 'movie_title', 'movie_tag', 'movie_year', 'match_score']]

    def multi_query_retrieve(self, queries: List[str], excluded_ids: List[int]) -> pd.DataFrame:
        """多查询融合检索（GPT4Rec核心策略）"""
        all_recommendations = []

        # 为每个查询检索电影（带进度条）
        for query in tqdm(queries, desc="基于查询检索电影"):
            matched = self.retrieve_movies(query, excluded_ids)
            if not matched.empty:
                # 添加查询来源标识
                matched['query_source'] = query
                all_recommendations.append(matched)

        # 合并所有检索结果，去重并排序
        if all_recommendations:
            combined = pd.concat(all_recommendations, ignore_index=True)
            # 去重（保留匹配分数最高的）
            combined = combined.sort_values('match_score', ascending=False).drop_duplicates('movie_id').reset_index(
                drop=True)
            # 取前N个推荐
            return combined.head(config.RECOMMENDATION_NUM)

        return pd.DataFrame()


# ===================== 主推荐流程（整合GPT4Rec全链路）=====================
class GPT4RecMovieRecommender:
    """GPT4Rec电影推荐器，整合查询生成和检索流程"""

    def __init__(self, config: GPT4RecConfig):
        self.config = config
        self.data_loader = MovieDataLoader(config)
        self.query_generator = LLMQueryGenerator(config)
        self.retriever = MovieRetriever(self.data_loader)

    def recommend(self, user_id: int) -> pd.DataFrame:
        """为指定用户生成推荐"""
        print(f"\n{'=' * 80}")
        print(f"🎬 GPT4Rec 电影推荐 - 用户ID: {user_id}")
        print(f"{'=' * 80}\n")

        # 1. 获取用户历史观看记录
        watched_movies, watched_ids = self.data_loader.get_user_history(user_id)
        if watched_movies.empty:
            print(f"❌ 用户 {user_id} 无观看记录，无法生成推荐")
            return pd.DataFrame()

        print(f"📜 用户历史观看记录（Top {self.config.USER_HISTORY_TOP_N}）：")
        for _, row in watched_movies.iterrows():
            year = row['movie_year'] if pd.notna(row['movie_year']) else '未知'
            print(f"  - {row['movie_title']} | 类型：{row['movie_tag']} | 年份：{year}")

        # 2. 生成多样化兴趣查询
        print(f"\n🔍 生成 {self.config.NUM_QUERIES} 个用户兴趣查询：")
        queries = self.query_generator.generate_queries(watched_movies)
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")

        # 3. 多查询融合检索
        print(f"\n🎯 基于查询检索电影（目标推荐 {self.config.RECOMMENDATION_NUM} 部）：")
        recommended_movies = self.retriever.multi_query_retrieve(queries, watched_ids)

        # 4. 输出最终结果
        print(f"\n{'=' * 80}")
        print("🏆 最终推荐结果：")
        print(f"{'=' * 80}")

        if not recommended_movies.empty:
            for idx, (_, row) in enumerate(recommended_movies.iterrows(), 1):
                year = row['movie_year'] if pd.notna(row['movie_year']) else '未知'
                print(f"  {idx}. {row['movie_title']}")
                print(f"     类型：{row['movie_tag']} | 年份：{year} | 匹配分数：{row['match_score']:.2f}")
                print(f"     来源查询：{row['query_source'][:50]}...")
                print()
        else:
            print("❌ 未找到匹配的推荐电影，建议调整查询生成策略或匹配阈值")

        return recommended_movies


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 初始化推荐器
    recommender = GPT4RecMovieRecommender(config)

    # 推荐示例（可修改user_id）
    target_user_id = 16
    recommender.recommend(user_id=target_user_id)
"""
GPT4Rec：基于生成式语言模型的个性化推荐框架
借助 LLM 生成用户兴趣查询，使用 BM25 搜索检索相关电影

实施步骤：
1. 数据预处理：加载 MovieLens 100K 数据，构建用户序列
2. 查询生成：使用 LLM API 根据用户历史生成兴趣查询
3. 物品检索：使用 BM25 根据查询检索相关电影
4. 评估：计算 Recall@K, Diversity@K, Coverage@K
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict
import pickle

# 添加项目根目录到路径，以便导入 MyToken
sys.path.append(r"C:\Users\13744\PycharmProjects\aitest")
import MyToken

warnings.filterwarnings('ignore')

# 数据路径 - 使用绝对路径
DATA_PATH = r"C:\Users\13744\PycharmProjects\aitest\Week17\03_推荐系统\M_ML-100K"

# Prompt 模板
PROMPT_TEMPLATE = """你是一个电影推荐专家。请分析用户历史观看的电影，生成 10 个不同的搜索查询来描述这个用户的兴趣偏好。

每个查询应该：
- 捕捉用户兴趣的不同方面（类型、演员、导演、主题等）
- 简洁明了，适合作为搜索关键词
- 用中文表达

用户历史观看的电影：
{movie_history}

请生成 10 个搜索查询，每行一个："""


# ============================================================================
# Step 1: 数据预处理
# ============================================================================

class DataPreprocessor:
    """数据预处理类：加载和准备 MovieLens 数据"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.user_sequences = None

    def load_data(self):
        """加载评分和电影数据"""
        print("正在加载数据...")
        self.ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            sep="::",
            header=None,
            engine='python'
        )
        self.ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

        self.movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            sep="::",
            header=None,
            engine='python',
            encoding="latin"
        )
        self.movies.columns = ["movie_id", "movie_title", "movie_tag"]

        print(f"加载数据完成：{len(self.ratings)} 条评分，{len(self.movies)} 部电影")

    def prepare_sequences(self, min_rating=3, max_len=15):
        """
        准备用户序列数据

        Args:
            min_rating: 最低评分视为正样本
            max_len: 序列最大长度
        """
        print("正在准备用户序列...")

        # 过滤正样本
        ratings_positive = self.ratings[self.ratings['rating'] >= min_rating].copy()

        # 按用户和时间排序
        ratings_positive = ratings_positive.sort_values(['user_id', 'timestamp'])

        # 构建用户序列
        user_sequences = {}
        for user_id in ratings_positive['user_id'].unique():
            user_movies = ratings_positive[ratings_positive['user_id'] == user_id]['movie_id'].tolist()

            # 去重并限制长度
            user_movies = list(dict.fromkeys(user_movies))  # 去重保持顺序
            if len(user_movies) > max_len:
                user_movies = user_movies[-max_len:]  # 保留最近的电影

            user_sequences[user_id] = user_movies

        self.user_sequences = user_sequences
        print(f"用户序列构建完成：{len(user_sequences)} 个用户")

    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """
        划分训练/验证/测试集

        Returns:
            train_sequences, val_sequences, test_sequences
        """
        print("正在划分数据集...")

        train_sequences = {}
        val_sequences = {}
        test_sequences = {}

        for user_id, movies in self.user_sequences.items():
            if len(movies) < 3:  # 至少需要3部电影
                continue

            # 按时间划分：前80%训练，中间10%验证，最后10%测试
            n = len(movies)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_sequences[user_id] = movies[:train_end]
            val_sequences[user_id] = movies[train_end:val_end]
            test_sequences[user_id] = movies[val_end:]

        print(f"数据集划分完成：")
        print(f"  训练集: {len(train_sequences)} 用户")
        print(f"  验证集: {len(val_sequences)} 用户")
        print(f"  测试集: {len(test_sequences)} 用户")

        return train_sequences, val_sequences, test_sequences

    def get_movie_titles(self, movie_ids: List[int]) -> List[str]:
        """获取电影标题列表"""
        titles = []
        for mid in movie_ids:
            movie = self.movies[self.movies['movie_id'] == mid]
            if len(movie) > 0:
                titles.append(movie.iloc[0]['movie_title'])
            else:
                titles.append(f"Movie {mid}")
        return titles


# ============================================================================
# Step 2: 查询生成（使用 LLM API）
# ============================================================================

class QueryGenerator:
    """查询生成器：使用 LLM API 生成用户兴趣查询"""

    def __init__(self, api_key: str, base_url: str, model: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.model = model
            self.use_api = True
            print(f"LLM API 初始化成功: {model}")
        except Exception as e:
            print(f"LLM API 初始化失败: {e}")
            print("将使用基于规则的查询生成作为后备方案")
            self.use_api = False
            self.client = None

    def generate_queries(self, movie_titles: List[str], num_queries: int = 10) -> List[str]:
        """
        生成用户兴趣查询

        Args:
            movie_titles: 用户历史观看的电影标题列表
            num_queries: 需要生成的查询数量

        Returns:
            查询列表
        """
        if not movie_titles:
            return []

        if self.use_api:
            return self._generate_with_llm(movie_titles, num_queries)
        else:
            return self._generate_with_rules(movie_titles, num_queries)

    def _generate_with_llm(self, movie_titles: List[str], num_queries: int) -> List[str]:
        """使用 LLM API 生成查询"""
        try:
            # 格式化 prompt
            movie_history = "\n".join([f"{i+1}. {title}" for i, title in enumerate(movie_titles)])
            prompt = PROMPT_TEMPLATE.format(movie_history=movie_history)

            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的电影推荐系统助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )

            # 解析响应
            content = response.choices[0].message.content.strip()
            queries = [q.strip() for q in content.split('\n') if q.strip()]

            # 确保返回足够的查询
            while len(queries) < num_queries and len(queries) > 0:
                queries.extend(queries)

            return queries[:num_queries]

        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            return self._generate_with_rules(movie_titles, num_queries)

    def _generate_with_rules(self, movie_titles: List[str], num_queries: int) -> List[str]:
        """
        基于规则生成查询（后备方案）
        使用电影类型和关键词提取
        """
        # 从电影标题中提取关键词（简化版）
        queries = []

        # 基础查询
        queries.append("科幻电影")
        queries.append("动作片")
        queries.append("经典电影")
        queries.append("剧情片")
        queries.append("喜剧电影")
        queries.append("惊悚片")
        queries.append("爱情电影")
        queries.append("冒险电影")
        queries.append("悬疑片")
        queries.append("热门电影")

        # 确保返回足够的查询
        while len(queries) < num_queries:
            queries.extend(queries)

        return queries[:num_queries]


# ============================================================================
# Step 3: 物品检索（BM25）
# ============================================================================

class BM25Retriever:
    """BM25 检索器：根据查询检索相关电影"""

    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        self.corpus = []
        self.movie_ids = []
        self.bm25 = None

        self._build_index()

    def _build_index(self):
        """构建 BM25 索引"""
        print("正在构建 BM25 索引...")

        # 准备语料库：电影标题 + 类型标签
        for _, row in self.movies_df.iterrows():
            movie_id = row['movie_id']
            title = row['movie_title']
            tags = row['movie_tag'] if pd.notna(row['movie_tag']) else ""

            # 组合标题和类型
            doc = f"{title} {tags}"
            self.corpus.append(doc.lower())
            self.movie_ids.append(movie_id)

        # 尝试使用 rank_bm25，如果不可用则使用简单实现
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("BM25 索引构建完成（使用 rank_bm25）")
        except ImportError:
            print("rank_bm25 不可用，将使用简单的 TF-IDF 检索")
            self._build_simple_index()

    def _build_simple_index(self):
        """构建简单的 TF-IDF 索引（后备方案）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        检索 top-k 相关电影

        Args:
            query: 搜索查询
            k: 返回电影数量

        Returns:
            [(movie_id, score), ...] 列表
        """
        # 如果是 BM25
        if hasattr(self, 'bm25') and self.bm25 is not None:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
        else:
            # 使用 TF-IDF
            query_vec = self.vectorizer.transform([query.lower()])
            scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][:k]

        results = [(self.movie_ids[i], float(scores[i])) for i in top_indices]
        return results

    def get_movie_info(self, movie_id: int) -> Dict:
        """获取电影信息"""
        movie = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if len(movie) > 0:
            return {
                'movie_id': int(movie.iloc[0]['movie_id']),
                'title': movie.iloc[0]['movie_title'],
                'tags': movie.iloc[0]['movie_tag'] if pd.notna(movie.iloc[0]['movie_tag']) else ""
            }
        return None


# ============================================================================
# Step 4: 评估指标
# ============================================================================

class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def recall_at_k(recommended_items: List[int], target_items: List[int], k: int) -> float:
        """
        计算 Recall@K

        Args:
            recommended_items: 推荐物品列表
            target_items: 目标物品列表
            k: top-k

        Returns:
            Recall@K 分数
        """
        top_k = recommended_items[:k]
        hits = len(set(top_k) & set(target_items))
        return hits / len(target_items) if target_items else 0.0

    @staticmethod
    def diversity_at_k(recommended_items: List[int], movies_df: pd.DataFrame, k: int) -> float:
        """
        计算 Diversity@K（基于类型的 Jaccard 相似度）

        Args:
            recommended_items: 推荐物品列表
            movies_df: 电影数据
            k: top-k

        Returns:
            Diversity@K 分数
        """
        top_k = recommended_items[:k]

        # 获取每个推荐电影的类型集合
        item_categories = []
        for item_id in top_k:
            movie = movies_df[movies_df['movie_id'] == item_id]
            if len(movie) > 0:
                tags = movie.iloc[0]['movie_tag']
                if pd.notna(tags):
                    categories = set(tags.split('|'))
                else:
                    categories = set()
            else:
                categories = set()
            item_categories.append(categories)

        # 计算平均 Jaccard 相似度
        if len(item_categories) < 2:
            return 0.0

        total_similarity = 0.0
        count = 0

        for i in range(len(item_categories)):
            for j in range(i + 1, len(item_categories)):
                intersection = len(item_categories[i] & item_categories[j])
                union = len(item_categories[i] | item_categories[j])
                similarity = intersection / union if union > 0 else 0.0
                total_similarity += similarity
                count += 1

        avg_similarity = total_similarity / count if count > 0 else 0.0
        diversity = 1.0 - avg_similarity

        return diversity

    @staticmethod
    def coverage_at_k(recommended_items: List[int], user_history: List[int],
                     movies_df: pd.DataFrame, k: int) -> float:
        """
        计算 Coverage@K（推荐物品覆盖用户历史类型的程度）

        Args:
            recommended_items: 推荐物品列表
            user_history: 用户历史物品列表
            movies_df: 电影数据
            k: top-k

        Returns:
            Coverage@K 分数
        """
        top_k = recommended_items[:k]

        # 获取用户历史的类型集合
        user_categories = set()
        for item_id in user_history:
            movie = movies_df[movies_df['movie_id'] == item_id]
            if len(movie) > 0:
                tags = movie.iloc[0]['movie_tag']
                if pd.notna(tags):
                    user_categories.update(tags.split('|'))

        if not user_categories:
            return 0.0

        # 获取推荐物品的类型集合
        rec_categories = set()
        for item_id in top_k:
            movie = movies_df[movies_df['movie_id'] == item_id]
            if len(movie) > 0:
                tags = movie.iloc[0]['movie_tag']
                if pd.notna(tags):
                    rec_categories.update(tags.split('|'))

        # 计算覆盖率
        intersection = len(user_categories & rec_categories)
        coverage = intersection / len(user_categories)

        return coverage


# ============================================================================
# Step 5: 主执行流程
# ============================================================================

class GPT4RecRecommender:
    """GPT4Rec 推荐系统主类"""

    def __init__(self, data_path: str, use_llm: bool = True,
                 api_key: str = None, base_url: str = None, model: str = None):
        # 数据预处理
        self.preprocessor = DataPreprocessor(data_path)
        self.preprocessor.load_data()
        self.preprocessor.prepare_sequences()

        # BM25 检索器
        self.retriever = BM25Retriever(self.preprocessor.movies)

        # 查询生成器
        if use_llm and api_key:
            self.query_generator = QueryGenerator(api_key, base_url, model)
        else:
            self.query_generator = QueryGenerator("", "", "")
            print("使用基于规则的查询生成")

    def recommend(self, user_history: List[int], num_queries: int = 10,
                 top_k: int = 20) -> Tuple[List[int], List[str]]:
        """
        为用户生成推荐

        Args:
            user_history: 用户历史观看的电影 ID 列表
            num_queries: 生成的查询数量
            top_k: 返回的推荐数量

        Returns:
            (推荐电影ID列表, 生成的查询列表)
        """
        # 获取电影标题
        movie_titles = self.preprocessor.get_movie_titles(user_history)

        # 生成查询
        queries = self.query_generator.generate_queries(movie_titles, num_queries)

        # 使用每个查询检索电影
        all_recommendations = {}
        for query in queries:
            results = self.retriever.retrieve(query, k=top_k // num_queries + 5)
            for movie_id, score in results:
                if movie_id not in user_history:  # 排除已观看的
                    if movie_id not in all_recommendations:
                        all_recommendations[movie_id] = []
                    all_recommendations[movie_id].append(score)

        # 排序并返回 top-k
        # 简单策略：平均分数
        ranked_movies = sorted(
            all_recommendations.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )

        recommended_ids = [mid for mid, _ in ranked_movies[:top_k]]

        return recommended_ids, queries

    def evaluate(self, test_sequences: Dict, num_users: int = None,
                 num_queries: int = 10) -> Dict:
        """
        评估推荐系统

        Args:
            test_sequences: 测试集用户序列
            num_users: 测试用户数量（None 表示全部）
            num_queries: 每个用户生成的查询数量

        Returns:
            评估结果字典
        """
        print(f"\n开始评估...")

        # 限制测试用户数量
        test_users = list(test_sequences.keys())
        if num_users:
            test_users = test_users[:num_users]

        print(f"测试用户数: {len(test_users)}")

        # 存储结果
        all_results = []

        for user_id in tqdm(test_users, desc="生成推荐"):
            user_movies = test_sequences[user_id]

            if len(user_movies) < 2:
                continue

            # 使用除最后一个电影作为历史，最后一个作为目标
            history = user_movies[:-1]
            target = user_movies[-1:]

            # 生成推荐
            recommended_ids, queries = self.recommend(history, num_queries=num_queries, top_k=40)

            all_results.append({
                'user_id': user_id,
                'history': history,
                'target': target,
                'recommended': recommended_ids,
                'queries': queries
            })

        # 计算评估指标
        k_values = [5, 10, 20, 40]
        metrics = {f'recall@{k}': [] for k in k_values}
        metrics.update({f'diversity@{k}': [] for k in k_values})
        metrics.update({f'coverage@{k}': [] for k in k_values})

        for result in all_results:
            for k in k_values:
                # Recall@K
                recall = MetricsCalculator.recall_at_k(
                    result['recommended'], result['target'], k
                )
                metrics[f'recall@{k}'].append(recall)

                # Diversity@K
                diversity = MetricsCalculator.diversity_at_k(
                    result['recommended'], self.preprocessor.movies, k
                )
                metrics[f'diversity@{k}'].append(diversity)

                # Coverage@K
                coverage = MetricsCalculator.coverage_at_k(
                    result['recommended'], result['history'],
                    self.preprocessor.movies, k
                )
                metrics[f'coverage@{k}'].append(coverage)

        # 计算平均值
        final_metrics = {}
        for key, values in metrics.items():
            final_metrics[key] = np.mean(values)

        return final_metrics, all_results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主执行函数"""
    import argparse

    parser = argparse.ArgumentParser(description='GPT4Rec 推荐系统')
    parser.add_argument('--test-users', type=int, default=20,
                       help='测试用户数量（默认 20，用于快速测试）')
    parser.add_argument('--num-queries', type=int, default=10,
                       help='每个用户生成的查询数量')
    parser.add_argument('--no-llm', action='store_true',
                       help='不使用 LLM API，使用基于规则的查询生成')

    args = parser.parse_args()

    print("=" * 60)
    print("GPT4Rec 推荐系统")
    print("=" * 60)

    # 从 MyToken.py 读取 API 配置
    # 默认使用阿里云通义千问
    api_key = getattr(MyToken, 'aliyun', None)
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    model = 'qwen-turbo'

    # 如果环境变量中有配置，则使用环境变量（便于覆盖）
    if os.environ.get('OPENAI_API_KEY'):
        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('OPENAI_BASE_URL', base_url)
        model = os.environ.get('OPENAI_MODEL', model)

    use_llm = not args.no_llm and api_key

    if use_llm:
        print(f"\n使用 LLM API: {model}")
        print(f"API 地址: {base_url}")
    else:
        print("\n使用基于规则的查询生成")

    # 初始化推荐系统
    recommender = GPT4RecRecommender(
        data_path=DATA_PATH,
        use_llm=use_llm,
        api_key=api_key if use_llm else None,
        base_url=base_url if use_llm else None,
        model=model if use_llm else None
    )

    # 划分数据集
    _, _, test_sequences = recommender.preprocessor.split_data()

    # 评估
    metrics, results = recommender.evaluate(
        test_sequences,
        num_users=args.test_users,
        num_queries=args.num_queries
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    for k in [5, 10, 20, 40]:
        print(f"\nTop-{k} 推荐性能:")
        print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
        print(f"  Diversity@{k}: {metrics[f'diversity@{k}']:.4f}")
        print(f"  Coverage@{k}:  {metrics[f'coverage@{k}']:.4f}")

    # 展示示例推荐
    print("\n" + "=" * 60)
    print("示例推荐")
    print("=" * 60)

    for i, result in enumerate(results[:3]):
        print(f"\n用户 {result['user_id']}:")
        print(f"  历史电影数量: {len(result['history'])}")

        print(f"  生成的查询:")
        for j, query in enumerate(result['queries'][:5], 1):
            print(f"    {j}. {query}")

        print(f"  目标电影:")
        target_movie = recommender.retriever.get_movie_info(result['target'][0])
        if target_movie:
            print(f"    {target_movie['title']}")

        print(f"  Top-5 推荐:")
        for j, movie_id in enumerate(result['recommended'][:5], 1):
            movie = recommender.retriever.get_movie_info(movie_id)
            if movie:
                is_hit = "✓" if movie_id in result['target'] else ""
                print(f"    {j}. {movie['title']} {is_hit}")

    # 保存结果
    output_file = "gpt4rec_results.json"

    # 转换 numpy 类型为 Python 原生类型
    def convert_to_serializable(obj):
        """将 numpy 类型和其他不可序列化对象转换为 JSON 可序列化类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # 准备可序列化的数据
    serializable_data = {
        'metrics': convert_to_serializable(metrics),
        'results': convert_to_serializable(results[:10])  # 只保存前10个用户的详细结果
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

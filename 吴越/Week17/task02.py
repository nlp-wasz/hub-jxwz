import pandas as pd
import numpy as py
from rank_bm25 import BM25Okapi
from typing import List,Dict
import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import warnings

import re

warnings.filterwarnings('ignore')

config={
    "model_path":"the path to model",
    "ratings_path":"the path to ratings",
    "movies_path":"the path to movies"
}

class GPT4Rec():
    def __init__(self,config):
        """

        Args:
            model_name:模型地址
        """
        self.tokenizer=GPT2Tokenizer.from_pretrained(config["model_path"])
        self.model=GPT2LMHeadModel.from_pretrained(config["model_path"])
        #加载评分数据
        self.ratings_df=pd.read_csv(config["ratings_path"],sep="::",header=None,engine="python")
        self.ratings_df.columns=["user_id","movie_id","rating","timestamp"]
        #
        self.movies_df = pd.read_csv(config["movies_path"], sep="::", header=None, engine="python")
        self.movies_df.columns = ["movie_id", "title", "genres"]

        self.bm25=None


    def search_with_bm25(self):
        documents = []

        for _, row in self.movies_df.iterrows():
            # 分离标题和类型
            title = row['title']
            genres = row['genres']

            # 类型拆分为单独的词
            genre_terms = genres.replace('|', ' ')

            enhanced_doc = f"{title} {genre_terms}"
            documents.append(enhanced_doc.lower())

        # 创建BM25
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_user_history(self, user_id: int, top_n: int = 5) -> List[str]:
        """
        获取用户的历史观看记录（按评分和时间排序）

        参数:
        user_id: 用户ID
        top_n: 返回的历史记录数量，默认为5

        返回:
        用户观看过的电影标题和类型列表，格式为 "标题 (类型1|类型2|...)"
        """
        if user_id not in self.ratings_df['user_id'].values:
            print(f"警告: 用户ID {user_id} 不存在于评分数据中")
            return []


        # 1. 获取用户的评分记录并排序
        user_ratings = (
            self.ratings_df[self.ratings_df['user_id'] == user_id]
            .sort_values(by=['rating', 'timestamp'], ascending=[False, False])
            .head(top_n)
        )

        if user_ratings.empty:
            print(f"用户 {user_id} 没有评分记录")
            return []

        # 2. 获取对应的电影信息（使用merge提高效率）
        history_movies = pd.merge(
            user_ratings[['movie_id', 'rating', 'timestamp']],
            self.movies_df[['movie_id', 'title', 'genres']],
            on='movie_id',
            how='left'
        )

        # 3. 创建格式化的电影描述
        def format_movie_info(row):
            return f"{row['title']} (评分: {row['rating']}, 类型: {row['genres']})"

        history_movie_result = history_movies.apply(format_movie_info, axis=1).tolist()

        return history_movie_result

    def GPT_generate_recommendtion_movie(self, user_history: List[str],query:str) -> str:
        """
        使用GPT生成搜索查询

        参数:
        user_history: 用户历史观看的电影列表

        返回:
        生成的搜索查询
        """
        # 构建提示词
        history_text = "\n".join([movie for movie in user_history])

        prompt = f"""你是一个电影推荐专家。推荐用户未来可能观看的电影。
    用户历史观看的电影：
    {history_text}
    基于上述用户历史观看电影，请推荐10个待选的电影，每行一个推荐
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
        # 解析成列表
        recommendations = self._parse_recommendations(query)
        return recommendations

    def _parse_recommendations(self, text: str) -> List[str]:
        """
        解析推荐文本为列表
        """
        recommendations = []

        # 按行分割
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for line in lines:
            # 移除数字编号和特殊字符
            line = re.sub(r'^\d+[\.、\)\]]\s*', '', line)  # 移除 "1.", "1)", "1]"
            line = re.sub(r'^[-\*•]\s*', '', line)  # 移除 "- ", "* ", "• "
            line = line.strip()

            if line and len(line) > 2:  # 过滤空行和过短的文本
                recommendations.append(line)

        # 限制返回数量
        return recommendations[:10]

    def get_similarity_movies(self,movies_list:list,top_n:int=5)->Dict[str,List[str]]:
        """
        根据电影列表获取每部电影的相似电影
        :param movies_list: 电影标题
        :param top_n:为每部电影返回的相似电影数量，默认为5
        :return: 字典，键为输入电影，值为相似电影列表
        """
        #存储结果
        similarity_movies = {}
        for movie_query in movies_list:
            query_tokens=movie_query.lower().split()
            #使用BM25获取相似度分数
            scores=self.bm25.get_scores(query_tokens)
            #获取top_n个最相似的电影
            top_indices=scores.argsort()[-top_n][::-1]

            results_df=self.movies_df.iloc[top_indices]
            results_df['bm25_score'] = scores[top_indices]

            similar_titles=[]
            for _,row in results_df.iterrows():
                movie_info=f"{row['title']} (类型: {row['genres']},分数: {row['bm25_score']})"
                similar_titles.append(movie_info)
            similarity_movies[movie_query]=similar_titles

        return similarity_movies







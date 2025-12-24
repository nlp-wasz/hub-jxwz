"""
GPT4Rec ：基于生成式语言模型的个性化推荐框架，借助 GPT2 生成未来的查询条件，使用搜索检索到相关的物品。
- 步骤1（生成查询条件）: 根据用户历史交互物品的文本信息（如商品标题），生成能够代表用户未来兴趣的、可读的"搜索查询"。
    Previously, the customer has bought: <标题1>. <标题2>... In the future, the customer wants to buy
- 步骤2（物品的检索）: 从整个物品库中检索出最相关的物品作为推荐候选
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import warnings
import requests
import json

warnings.filterwarnings('ignore')

# 读取数据
try:
    ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

    movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
    movies.columns = ["movie_id", "movie_title", "movie_genres"]  # 修正列名
except FileNotFoundError:
    # 如果文件不存在，创建模拟数据用于演示
    print("数据文件未找到，创建模拟数据用于演示...")

    # 创建模拟数据
    n_users = 100
    n_movies = 50
    n_ratings = 1000

    np.random.seed(42)
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    movie_ids = np.random.randint(1, n_movies + 1, n_ratings)
    ratings_data = np.random.randint(1, 6, n_ratings)  # 评分1-5
    timestamps = np.random.randint(800000000, 900000000, n_ratings)

    ratings = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings_data,
        'timestamp': timestamps
    })

    # 创建模拟电影数据
    movie_titles = [f"电影_{i}_标题" for i in range(1, n_movies + 1)]
    movie_genres = [f"类型_{i % 5}" for i in range(1, n_movies + 1)]
    movies = pd.DataFrame({
        'movie_id': range(1, n_movies + 1),
        'movie_title': movie_titles,
        'movie_genres': movie_genres
    })


class GPT4RecDataset(Dataset):
    """GPT4Rec数据集类"""

    def __init__(self, ratings, movies, seq_len=10):
        self.ratings = ratings
        self.movies = movies
        self.seq_len = seq_len
        self.user_movie_history = self._build_user_history()

    def _build_user_history(self):
        """构建用户电影历史记录"""
        user_movie_history = {}
        # 按时间排序
        sorted_ratings = self.ratings.sort_values(['user_id', 'timestamp'])

        for user_id in sorted_ratings['user_id'].unique():
            user_ratings = sorted_ratings[sorted_ratings['user_id'] == user_id]
            movie_ids = user_ratings['movie_id'].tolist()
            user_movie_history[user_id] = movie_ids

        return user_movie_history

    def __len__(self):
        return len(self.user_movie_history)

    def __getitem__(self, idx):
        user_id = list(self.user_movie_history.keys())[idx]
        movie_history = self.user_movie_history[user_id]

        # 如果历史太短，需要扩展
        if len(movie_history) < 2:
            # 如果用户只有一条记录，无法预测下一个，返回空
            return torch.tensor([0] * self.seq_len), torch.tensor([0])

        # 准备序列
        if len(movie_history) <= self.seq_len:
            # 如果历史长度小于序列长度，填充0
            seq = [0] * (self.seq_len - len(movie_history)) + movie_history[:-1]
            target = movie_history[-1]
        else:
            # 如果历史长度大于序列长度，取最后seq_len个
            seq = movie_history[-self.seq_len - 1:-1]  # 前seq_len个作为输入
            target = movie_history[-1]  # 最后一个作为目标

        return torch.tensor(seq, dtype=torch.long), torch.tensor([target], dtype=torch.long)


class GPT4RecModel(nn.Module):
    """GPT4Rec模型实现"""

    def __init__(self, num_items, embedding_dim=64, nhead=8, num_layers=2, seq_len=10):
        super(GPT4RecModel, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # 项目嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        # 位置嵌入层
        self.position_embedding = nn.Embedding(seq_len + 1, embedding_dim)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # 项目嵌入
        item_embeds = self.item_embedding(x)  # [batch_size, seq_len, embedding_dim]

        # 位置嵌入
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        pos_embeds = self.position_embedding(positions)

        # 组合嵌入
        embeds = item_embeds + pos_embeds
        embeds = self.dropout(embeds)

        # Transformer处理
        output = self.transformer(embeds)

        # 取序列最后一个位置的输出用于预测
        last_output = output[:, -1, :]  # [batch_size, embedding_dim]

        # 输出层
        logits = self.fc_out(last_output)  # [batch_size, num_items + 1]

        return logits


class GPT4Rec:
    """GPT4Rec推荐系统主类"""

    def __init__(self, seq_len=10, embedding_dim=64, nhead=8, num_layers=2, baichuan_api_key=None):
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.model = None
        self.movies = movies
        self.ratings = ratings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baichuan_api_key = "sk-6a30a43627f14a61a0b065d72a7c91f9"  # 百炼API密钥

    def prepare_data(self):
        """准备训练数据"""
        dataset = GPT4RecDataset(self.ratings, self.movies, seq_len=self.seq_len)
        # 这里简化处理，使用DataLoader
        return dataset

    def train(self, epochs=10, batch_size=32, lr=0.001):
        """训练模型"""
        dataset = self.prepare_data()

        # 为简化，这里直接使用所有数据，实际中应该划分训练/测试集
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_items = self.movies['movie_id'].max()
        self.model = GPT4RecModel(
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            seq_len=self.seq_len
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充的0
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for seq, target in dataloader:
                seq, target = seq.to(self.device), target.to(self.device).squeeze()

                # 确保target是正确的形状
                if target.dim() == 1:
                    target = target.squeeze()

                optimizer.zero_grad()
                output = self.model(seq)

                # 调整target形状以匹配output
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    def recommend_for_user(self, user_id, top_k=10):
        """为用户推荐电影"""
        if self.model is None:
            print("模型未训练，请先训练模型")
            return []

        # 获取用户历史
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('timestamp')
        history_movie_ids = user_ratings['movie_id'].tolist()

        # 构建序列
        if len(history_movie_ids) < 1:
            print(f"用户 {user_id} 没有足够的历史记录")
            return []

        # 取最近的seq_len个交互
        seq = history_movie_ids[-self.seq_len:]
        # 如果序列长度不足，用0填充
        if len(seq) < self.seq_len:
            seq = [0] * (self.seq_len - len(seq)) + seq

        seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(seq_tensor)
            # 获取top_k个最可能的项目
            _, top_k_items = torch.topk(output, top_k)
            top_k_items = top_k_items.cpu().numpy()[0]

        # 获取推荐的电影信息
        recommended_movies = []
        for movie_id in top_k_items:
            if movie_id != 0:  # 忽略填充项
                movie_info = self.movies[self.movies['movie_id'] == movie_id]
                if not movie_info.empty:
                    recommended_movies.append({
                        'movie_id': movie_id,
                        'movie_title': movie_info.iloc[0]['movie_title'],
                        'movie_genres': movie_info.iloc[0]['movie_genres']
                    })

        return recommended_movies

    def call_baichuan_api(self, prompt):
        """调用百炼API生成查询"""
        if not self.baichuan_api_key:
            print("未提供百炼API密钥，使用TF-IDF方法作为备选")
            return None

        # 百炼API调用示例 (这里使用阿里云百炼大模型API)
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        headers = {
            'Authorization': f'Bearer {self.baichuan_api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "qwen-max",  # 使用通义千问模型
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 200
            }
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                if 'output' in result and 'text' in result['output']:
                    return result['output']['text']
            else:
                print(f"API调用失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            print(f"API调用异常: {e}")
            return None

    def generate_query_based_recommendation(self, user_id, top_k=10):
        """基于查询生成的推荐（使用百炼API）"""
        # 获取用户历史观看的电影
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('timestamp', ascending=False).head(5)  # 取最近5个
        history_movie_ids = user_ratings['movie_id'].tolist()

        history_movies = self.movies[self.movies['movie_id'].isin(history_movie_ids)]
        history_titles = history_movies['movie_title'].tolist()

        # 打印用户历史
        print("用户历史观看电影:")
        for title in history_titles:
            print(f"  - {title}")

        # 构建提示词给大模型
        if history_titles:
            prompt = f"""
            你是一个电影推荐专家，请基于用户的历史观看记录，生成一个描述用户可能感兴趣的新电影的查询语句。
            用户历史观看的电影：{', '.join(history_titles)}
            
            请生成一个描述用户可能感兴趣的下一部电影的查询语句，例如：
            "用户可能喜欢{history_titles[0]}类型的电影，接下来可能会对[电影类型]、[主题]、[风格]的电影感兴趣"
            
            请简洁明了地描述用户可能感兴趣的电影特征：
            """

            # 调用百炼API生成查询
            generated_query = self.call_baichuan_api(prompt)

            if generated_query:
                print(f"\nAI生成的用户兴趣描述: {generated_query}")

                # 使用生成的查询进行电影推荐
                # 这里我们使用TF-IDF来匹配与AI生成描述最相似的电影
                all_movies_text = self.movies['movie_title'] + " " + self.movies['movie_genres']

                # 创建TF-IDF向量
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(all_movies_text.astype(str))

                # 将AI生成的查询向量化
                query_vector = tfidf.transform([generated_query])

                # 计算相似度
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

                # 获取最相似的电影索引
                top_indices = similarities.argsort()[-top_k - 5:][::-1]  # 多取几个避免重复

                # 过滤掉用户已经看过的电影
                recommended_movies = []
                for idx in top_indices:
                    movie_id = self.movies.iloc[idx]['movie_id']
                    if movie_id not in history_movie_ids and len(recommended_movies) < top_k:
                        recommended_movies.append({
                            'movie_id': movie_id,
                            'movie_title': self.movies.iloc[idx]['movie_title'],
                            'movie_genres': self.movies.iloc[idx]['movie_genres']
                        })

                print("\n基于AI生成查询的推荐电影:")
                return recommended_movies
            else:
                print("\nAPI调用失败，使用TF-IDF方法作为备选...")

        # 如果API调用失败或没有历史记录，使用TF-IDF方法
        if len(history_movie_ids) > 0:
            all_movies_text = self.movies['movie_title'] + " " + self.movies['movie_genres']

            # 创建TF-IDF向量
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(all_movies_text.astype(str))

            # 计算用户历史电影的平均向量
            history_indices = [i for i, movie_id in enumerate(self.movies['movie_id']) if movie_id in history_movie_ids]
            if history_indices:
                history_vector = tfidf_matrix[history_indices].mean(axis=0)
                history_vector = np.asarray(history_vector).flatten()

                # 计算相似度
                similarities = cosine_similarity([history_vector], tfidf_matrix).flatten()

                # 获取最相似的电影索引
                top_indices = similarities.argsort()[-top_k - 5:][::-1]  # 多取几个避免重复

                # 过滤掉用户已经看过的电影
                recommended_movies = []
                for idx in top_indices:
                    movie_id = self.movies.iloc[idx]['movie_id']
                    if movie_id not in history_movie_ids and len(recommended_movies) < top_k:
                        recommended_movies.append({
                            'movie_id': movie_id,
                            'movie_title': self.movies.iloc[idx]['movie_title'],
                            'movie_genres': self.movies.iloc[idx]['movie_genres']
                        })

                print("\n基于TF-IDF的推荐电影:")
                return recommended_movies

        return []


def main():
    """主函数 - 演示GPT4Rec的使用"""
    print("初始化GPT4Rec推荐系统...")

    # gpt4rec = GPT4Rec(seq_len=10, embedding_dim=32, nhead=4, num_layers=2, baichuan_api_key="your_baichuan_api_key_here")
    gpt4rec = GPT4Rec(seq_len=10, embedding_dim=32, nhead=4, num_layers=2)  # 不使用API密钥

    print("开始训练模型...")
    # 训练模型（使用较小的参数以加快训练）
    gpt4rec.train(epochs=5, batch_size=16, lr=0.001)

    print("\n训练完成！")

    # 获取一个用户进行推荐演示
    if not ratings.empty:
        sample_user_id = ratings['user_id'].iloc[0]  # 取第一个用户
        print(f"\n为用户 {sample_user_id} 生成推荐...")

        # 方法1: 基于序列预测的推荐
        recommendations = gpt4rec.recommend_for_user(sample_user_id, top_k=5)
        print("\n基于序列预测的推荐结果:")
        if recommendations:
            for i, movie in enumerate(recommendations, 1):
                print(f"{i}. {movie['movie_title']} (类型: {movie['movie_genres']})")
        else:
            print("未找到推荐结果")

        # 方法2: 基于查询生成的推荐（使用百炼API或TF-IDF备选）
        query_based_recs = gpt4rec.generate_query_based_recommendation(sample_user_id, top_k=5)
        if query_based_recs:
            for i, movie in enumerate(query_based_recs, 1):
                print(f"{i}. {movie['movie_title']} (类型: {movie['movie_genres']})")
        else:
            print("未找到推荐结果")
    else:
        print("没有可用的用户数据进行推荐")


if __name__ == "__main__":
    main()

# 04_GPT4Rec.py
import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 1. 加载数据
DATA_DIR = "../预习/03_推荐系统/M_ML-100K"

# 确保数据路径存在
if not os.path.exists(os.path.join(DATA_DIR, "ratings.dat")):
    raise FileNotFoundError("请确保 M_ML-100K/ratings.dat 存在")

ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"),
    sep="::",
    header=None,
    engine='python',
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    header=None,
    engine='python',
    encoding="latin1",
    names=["movie_id", "movie_title", "movie_tag"]
)

# 合并 ratings 和 movies
df = ratings.merge(movies, on="movie_id")


# 2. 构建用户历史（按时间排序，取最近 N 部）
def get_user_history(user_id, top_k=10):
    user_data = df[df["user_id"] == user_id].sort_values("timestamp")
    titles = user_data["movie_title"].tolist()[-top_k:]  # 最近的 top_k 部
    return titles


# 示例：取第一个用户
sample_user = df["user_id"].iloc[0]
history_titles = get_user_history(sample_user, top_k=5)
print(f"用户 {sample_user} 的历史观看：")
for t in history_titles:
    print(f"  - {t}")

# 3. 构建 Prompt（英文，中文效果差，乱七八糟）
PROMPT_TEMPLATE = """You are a movie recommendation expert. Based on the user's previously watched movies, recommend movies they might like in the future.

Previously watched movies:
{history}

Please generate 10 candidate movie descriptions or titles that the user may be interested in. One per line:"""

prompt = PROMPT_TEMPLATE.format(history="\n".join([f"- {t}" for t in history_titles]))

# 4. 加载 GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 设置 pad_token（GPT-2 默认无 pad_token）
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# 5. 生成推荐描述（模拟 查询）
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

# 生成参数
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=200,  # 生成最多 200 个新 token
    num_return_sequences=1,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 提取生成部分（去掉 prompt）
gen_part = generated_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

print("\nGPT4Rec 生成的推荐描述（模拟搜索查询）")
print(gen_part)

# 可选：按行分割（假设每行一个推荐）
recommendations = [line.strip() for line in gen_part.split("\n") if line.strip()][:10]
print("\n解析出的 10 条推荐：")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

# BM25 检索
from rank_bm25 import BM25Okapi
import jieba  # 中文需分词，英文可用空格 split

# 获取用户已观看的电影标题（用于过滤）
watched_titles = set(df[df["user_id"] == sample_user]["movie_title"].str.lower())

# 构建物品库（所有电影标题）
corpus = movies["movie_title"].tolist()
tokenized_corpus = [title.lower().split() for title in corpus]  # 英文简单分词（MovieLens 是英文数据集）
bm25 = BM25Okapi(tokenized_corpus)

# 存储每条 query 的检索结果（用于调试或展示）
print("\n各生成 Query 的 BM25 检索示例（Top-3）")
all_scores = []
for i, query in enumerate(recommendations, 1):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # 展示当前 query 的 top-3 检索结果
    top3_idx = (-scores).argsort()[:3]
    top3_movies = [corpus[idx] for idx in top3_idx]
    print(f"Query {i}: \"{query}\"")
    for j, movie in enumerate(top3_movies, 1):
        print(f"  → {j}. {movie}")
    all_scores.append(scores)

# 融合所有 query 的 BM25 得分（简单求和）
final_scores = sum(all_scores)  # shape: (len(corpus),)

# 创建 (score, index) 列表并排序
scored_items = [(final_scores[i], i) for i in range(len(corpus))]
scored_items.sort(key=lambda x: x[0], reverse=True)

# 过滤：去重 + 排除已观看 + 取 top 10
seen = set()
final_recommendations = []

for score, idx in scored_items:
    movie_title = corpus[idx]
    if movie_title.lower() in watched_titles:
        continue
    if movie_title in seen:
        continue
    seen.add(movie_title)
    final_recommendations.append((movie_title, score))
    if len(final_recommendations) >= 10:
        break

# 输出最终推荐结果
print(f"为用户 {sample_user} 生成的最终 Top-10 推荐（GPT4Rec + BM25）")
for rank, (title, score) in enumerate(final_recommendations, 1):
    print(f"{rank:2d}. {title}  (BM25 融合得分: {score:.3f})")

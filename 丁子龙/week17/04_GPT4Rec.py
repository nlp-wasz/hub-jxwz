"""
GPT4Rec ：基于生成式语言模型的个性化推荐框架，借助 GPT2 生成未来的查询条件，使用搜索检索到相关的物品。
- 步骤1（生成查询条件）: 根据用户历史交互物品的文本信息（如商品标题），生成能够代表用户未来兴趣的、可读的“搜索查询”。
    Previously, the customer has bought: <标题1>. <标题2>... In the future, the customer wants to buy
- 步骤2（物品的检索）: 从整个物品库中检索出最相关的物品作为推荐候选
"""
from pprint import pprint
import torch
import pandas as pd
from transformers import (AutoTokenizer,AutoModelForCausalLM,

                          GPT2Tokenizer,GPT2PreTrainedModel,
                          GPT2Model,GPT2LMHeadModel)
ratings = pd.read_csv("./Multimodal_Datasets/M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]
# save ratings to csv
ratings.to_csv("ratings.csv", index=False)

movies = pd.read_csv("./Multimodal_Datasets/M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]
movies.to_csv("movies.csv", index=False)

users = pd.read_csv("./Multimodal_Datasets/M_ML-100K/user.dat", sep="::", header=None, engine='python')

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{movies}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""
# ✅ 推荐用英文 prompt（GPT-2 原生支持）
PROMPT_TEMPLATE = """You are a movie recommender system. Predict 10 diverse and realistic search queries that the user might type to find new movies, based on their watch history.

Watch history (movie titles):
{movies}
Generate exactly 10 concise search queries (one per line, numbered 1. to 10.):
"""

PROMPT_TEMPLATE = """You are a movie recommender system. Based on the user's watch history, generate 10 NEW and DIVERSE recommendations for movies they have NOT watched yet.

❗ STRICT RULES:
- DO NOT mention any movie titles from the watch history.
- DO NOT repeat the same idea.
- recommendations must be short phrases (3-8 words), like "sci-fi heist movie" or "romantic drama in Paris".

Watch history (DO NOT REPEAT THESE):
{movies}

Generate 10 recommendation (numbered 1. to 10.):
1."""

model_path = "/home/dzl/baDouNLP/week/openai-community/gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
# pprint(model.config)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in encoded_input.items()}

    # 生成文本
with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=128,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(output_text)

def generate_next_movies(text: str):
    encoded_input = tokenizer(text, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in encoded_input.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return output_text

history_by_userids = ratings.groupby("user_id")
print(history_by_userids)
historys = []
for user_id, history in history_by_userids:
    # print(user_id)
    history = history.sort_values("timestamp", ascending=False)
    # print(history)
    batch_size = 15
    for start in range(0, len(history), batch_size):
        end = start + batch_size
        user_historys = history.iloc[start:end]
        historys.append(user_historys)
datasets = []
for user_historys in historys:
    movies_text = []
    idx = 0
    for id,item in user_historys.iterrows():
        user_id,movie_id,rating,timestamp = item
        movie_title = movies[movies["movie_id"] == movie_id]["movie_title"].values[0]
        movie_tag = movies[movies["movie_id"] == movie_id]["movie_tag"].values[0]
        text = f" {idx+1}:movie title is {movie_title}，movie tag is{movie_tag}\n"
        movies_text.append(text)
        idx += 1
    prompt_text = ",".join(movies_text[:-1])
    prompt_text = PROMPT_TEMPLATE.format(movies=prompt_text)
    datasets.append({"history":prompt_text,"prediction":movies_text[-1]})

for dataset in datasets:
    print("*"*80)
    print(dataset["prediction"])
    prediction = generate_next_movies(dataset["history"])
    print("*" * 80)
    print(prediction)
    if prediction == dataset["prediction"]:
        print("预测正确")
    else:
        print("预测错误")




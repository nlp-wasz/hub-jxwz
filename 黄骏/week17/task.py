"""
GPT4Rec Demo

论文 GPT4Rec 的核心思想：把推荐拆成「生成 Query」 + 「搜索检索」两步。

- 步骤1（生成 Query）:
  根据用户历史交互物品的文本信息（这里用电影 title + genre），生成能代表用户未来兴趣的查询语句（可解释）。
- 步骤2（搜索检索）:
  把 Query 输入搜索引擎（这里实现 BM25），从全量物品库检索 Top-K 作为推荐候选，并做多 Query 合并以提升多样性。

本脚本默认离线运行：用启发式方法生成多条 Query；也支持可选调用 OpenAI 兼容接口生成 Query（需要网络和 API Key）。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\(\d{4}\)\s*$")
_LEADING_ENUM_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.\)、)]|\(?\d+\)?[\.\)、)])\s*")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[’']", "", text)
    return _TOKEN_RE.findall(text)


def clean_title(title: str) -> str:
    return _YEAR_RE.sub("", str(title)).strip()


def strip_line_prefix(line: str) -> str:
    line = line.strip()
    line = _LEADING_ENUM_RE.sub("", line)
    return line.strip().strip('"').strip("'").strip()


@dataclass(frozen=True)
class GeneratedQuery:
    text: str
    score: float = 1.0


class BM25:
    def __init__(self, tokenized_corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        if not tokenized_corpus:
            raise ValueError("BM25 corpus is empty")

        self.k1 = float(k1)
        self.b = float(b)

        self.N = len(tokenized_corpus)
        self.doc_len: List[int] = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / max(1, self.N)

        self.df: Counter[str] = Counter()
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for doc_idx, doc in enumerate(tokenized_corpus):
            tf = Counter(doc)
            for term, freq in tf.items():
                self.df[term] += 1
                self.postings[term].append((doc_idx, freq))

        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            # 常用 BM25 IDF 变体：log(1 + (N - df + 0.5)/(df + 0.5))
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, freq in self.postings.get(term, []):
                dl = self.doc_len[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (freq * (self.k1 + 1.0)) / denom

        if not scores:
            return []

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def load_movielens_100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")

    ratings = pd.read_csv(ratings_path, sep="::", header=None, engine="python")
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

    movies = pd.read_csv(movies_path, sep="::", header=None, engine="python", encoding="latin")
    movies.columns = ["movie_id", "movie_title", "movie_tag"]

    return ratings, movies


def build_movie_documents(movies: pd.DataFrame) -> Tuple[List[str], List[int], Dict[int, int]]:
    docs: List[str] = []
    doc_idx_to_movie_id: List[int] = []
    movie_id_to_doc_idx: Dict[int, int] = {}

    for i, row in movies.iterrows():
        movie_id = int(row["movie_id"])
        title = str(row["movie_title"])
        tag = str(row.get("movie_tag", ""))
        tag_text = tag.replace("|", " ")
        doc_text = f"{title} {tag_text}"
        docs.append(doc_text)
        movie_id_to_doc_idx[movie_id] = len(doc_idx_to_movie_id)
        doc_idx_to_movie_id.append(movie_id)

    return docs, doc_idx_to_movie_id, movie_id_to_doc_idx


def pick_user_history(
    ratings: pd.DataFrame,
    min_rating: float,
    history_len: int,
    user_id: int | None,
    seed: int,
) -> Tuple[int, List[int], int]:
    df = ratings.copy()
    df = df[df["rating"] >= min_rating].sort_values(["user_id", "timestamp"])

    user_to_seq = df.groupby("user_id")["movie_id"].apply(list).to_dict()
    candidates = [u for u, seq in user_to_seq.items() if len(seq) >= max(2, history_len + 1)]
    if not candidates:
        raise ValueError("No user has enough interactions after filtering; try lower --min_rating or --history_len.")

    rng = random.Random(seed)
    if user_id is None:
        user_id = int(rng.choice(candidates))
    else:
        if user_id not in user_to_seq:
            raise ValueError(f"user_id={user_id} not found in ratings after filtering.")
        if len(user_to_seq[user_id]) < max(2, history_len + 1):
            raise ValueError(f"user_id={user_id} has too few interactions; try smaller --history_len.")

    seq = user_to_seq[user_id]
    target = int(seq[-1])
    if history_len <= 0:
        history = [int(x) for x in seq[:-1]]
    else:
        history = [int(x) for x in seq[-(history_len + 1) : -1]]
    return user_id, history, target


def generate_queries_offline(history_movies: pd.DataFrame, num_queries: int, seed: int) -> List[GeneratedQuery]:
    if history_movies.empty:
        return []

    genre_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()

    for _, row in history_movies.iterrows():
        title = clean_title(row["movie_title"])
        genres = str(row.get("movie_tag", "")).split("|") if pd.notna(row.get("movie_tag")) else []
        for g in genres:
            g = g.strip()
            if g:
                genre_counter[g.lower()] += 1

        for tok in tokenize(title):
            if tok in STOPWORDS:
                continue
            if tok.isdigit():
                continue
            if len(tok) <= 2:
                continue
            keyword_counter[tok] += 1

    top_genres = [g for g, _ in genre_counter.most_common(6)]
    top_keywords = [w for w, _ in keyword_counter.most_common(10)]
    if not top_genres and not top_keywords:
        return [GeneratedQuery(text="movie") for _ in range(num_queries)]

    rng = random.Random(seed)
    queries: List[GeneratedQuery] = []
    used = set()

    for i in range(max(1, num_queries)):
        num_g = 1 if len(top_genres) == 1 else rng.choice([1, 2, 3])
        num_k = rng.choice([0, 1, 2])

        genres = []
        if top_genres:
            start = i % len(top_genres)
            genres = [top_genres[(start + j) % len(top_genres)] for j in range(min(num_g, len(top_genres)))]

        keywords = []
        if top_keywords and num_k > 0:
            keywords = rng.sample(top_keywords, k=min(num_k, len(top_keywords)))

        q_tokens = [*genres, *keywords]
        q = " ".join([t for t in q_tokens if t]).strip()
        if not q:
            q = "movie"

        q_norm = " ".join(tokenize(q))
        if q_norm in used:
            q = f"{q} {rng.choice(top_genres) if top_genres else ''}".strip()
            q_norm = " ".join(tokenize(q))
        used.add(q_norm)

        score = 0.0
        for g in genres:
            score += genre_counter.get(g, 0)
        for w in keywords:
            score += keyword_counter.get(w, 0) * 0.5
        score += 1e-3 * (num_queries - i)

        queries.append(GeneratedQuery(text=q, score=float(score)))

    return queries[:num_queries]


def generate_queries_with_llm(history_titles: Sequence[str], num_queries: int, model: str) -> List[GeneratedQuery]:
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("openai package is not installed; install it or run without --use_llm.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; export it or run without --use_llm.")

    base_url = os.getenv("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    history_text = "\n".join([f"- {t}" for t in history_titles])
    prompt = f"""
You are a movie recommender system. Given the user's watched history, write {num_queries} diverse search queries
that represent the user's future interests. Each query should be a single line of short keywords (English),
preferably using MovieLens-style genres like Action, Comedy, Drama, Romance, Thriller, Sci-Fi, Animation, etc.

Rules:
- Output ONLY the {num_queries} queries, one per line.
- No numbering, no extra text.

Watched history:
{history_text}
""".strip()

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    content = completion.choices[0].message.content or ""

    lines = [strip_line_prefix(x) for x in content.splitlines()]
    lines = [x for x in lines if x]
    if not lines:
        raise RuntimeError("LLM returned empty queries.")

    queries: List[GeneratedQuery] = []
    for i, line in enumerate(lines[:num_queries]):
        queries.append(GeneratedQuery(text=line, score=1.0 - i * 1e-3))
    return queries


def merge_multi_query_results(
    query_to_results: Sequence[Tuple[GeneratedQuery, List[Tuple[int, float]]]],
    doc_idx_to_movie_id: Sequence[int],
    history_movie_ids: Iterable[int],
    top_k: int,
) -> List[Tuple[int, float]]:
    history_set = set(int(x) for x in history_movie_ids)
    picked: List[Tuple[int, float]] = []
    picked_set: set[int] = set()

    if top_k <= 0:
        return []

    queries_sorted = sorted(query_to_results, key=lambda x: x[0].score, reverse=True)
    per_query = max(1, top_k // max(1, len(queries_sorted)))

    # 先按 query 分配名额（类似论文的 K/m 策略）
    for q, results in queries_sorted:
        if len(picked) >= top_k:
            break
        added = 0
        for doc_idx, score in results:
            movie_id = int(doc_idx_to_movie_id[doc_idx])
            if movie_id in history_set:
                continue
            if movie_id in picked_set:
                continue
            picked.append((movie_id, float(score)))
            picked_set.add(movie_id)
            added += 1
            if added >= per_query or len(picked) >= top_k:
                break

    # 不足则按全局分数补齐（去重）
    if len(picked) < top_k:
        pool: Dict[int, float] = {}
        for _, results in queries_sorted:
            for doc_idx, score in results:
                movie_id = int(doc_idx_to_movie_id[doc_idx])
                if movie_id in history_set or movie_id in picked_set:
                    continue
                pool[movie_id] = max(pool.get(movie_id, 0.0), float(score))

        for movie_id, score in sorted(pool.items(), key=lambda x: x[1], reverse=True):
            picked.append((movie_id, score))
            picked_set.add(movie_id)
            if len(picked) >= top_k:
                break

    return picked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT4Rec demo: multi-query generation + BM25 retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--user_id", type=int, default=None, help="指定 user_id；默认随机抽取")
    parser.add_argument("--history_len", type=int, default=10, help="使用多少条历史（用于生成 Query）")
    parser.add_argument("--top_k", type=int, default=10, help="最终推荐 Top-K")
    parser.add_argument("--num_queries", type=int, default=10, help="生成 Query 数（multi-query）")
    parser.add_argument("--min_rating", type=float, default=3.0, help="过滤低分交互（>= min_rating）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_llm", action="store_true", help="使用 LLM 生成 Query（需要 OPENAI_API_KEY）")
    parser.add_argument("--llm_model", type=str, default="qwen-plus", help="OpenAI 兼容模型名（如 qwen-plus）")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "M_ML-100K")

    ratings, movies = load_movielens_100k(data_dir)

    docs, doc_idx_to_movie_id, _ = build_movie_documents(movies)
    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25(tokenized_docs, k1=1.5, b=0.75)

    user_id, history_ids, target_id = pick_user_history(
        ratings=ratings,
        min_rating=args.min_rating,
        history_len=args.history_len,
        user_id=args.user_id,
        seed=args.seed,
    )

    movies_by_id = movies.set_index("movie_id")
    history_movies = movies_by_id.loc[history_ids].reset_index()
    target_title = movies_by_id.loc[target_id]["movie_title"] if target_id in movies_by_id.index else str(target_id)

    history_titles = [str(x) for x in history_movies["movie_title"].tolist()]

    queries: List[GeneratedQuery]
    if args.use_llm:
        try:
            queries = generate_queries_with_llm(history_titles, num_queries=args.num_queries, model=args.llm_model)
        except Exception as e:
            print(f"[WARN] LLM query generation failed, fallback to offline generator: {e}")
            queries = generate_queries_offline(history_movies, num_queries=args.num_queries, seed=args.seed)
    else:
        queries = generate_queries_offline(history_movies, num_queries=args.num_queries, seed=args.seed)

    query_to_results: List[Tuple[GeneratedQuery, List[Tuple[int, float]]]] = []
    for q in queries:
        results = bm25.search(q.text, top_n=max(args.top_k * 5, 50))
        query_to_results.append((q, results))

    recs = merge_multi_query_results(
        query_to_results=query_to_results,
        doc_idx_to_movie_id=doc_idx_to_movie_id,
        history_movie_ids=history_ids,
        top_k=args.top_k,
    )

    print("=" * 80)
    print(f"user_id={user_id} | history_len={len(history_ids)} | target={target_title}")
    print("-" * 80)
    print("History:")
    for t in history_titles:
        print(f"  - {t}")
    print("-" * 80)
    print("Generated Queries:")
    for q in sorted(queries, key=lambda x: x.score, reverse=True):
        print(f"  ({q.score:.3f}) {q.text}")
    print("-" * 80)
    print(f"Top-{args.top_k} Recommendations (BM25 retrieval, excluding history):")
    hit = False
    for rank, (movie_id, score) in enumerate(recs, start=1):
        title = movies_by_id.loc[movie_id]["movie_title"] if movie_id in movies_by_id.index else str(movie_id)
        genres = movies_by_id.loc[movie_id]["movie_tag"] if movie_id in movies_by_id.index else ""
        flag = ""
        if int(movie_id) == int(target_id):
            flag = "  <-- HIT(target)"
            hit = True
        print(f"{rank:>2}. {title} | {genres} | score={score:.4f}{flag}")

    print("-" * 80)
    print(f"Recall@{args.top_k}: {1 if hit else 0}")
    print("=" * 80)


if __name__ == "__main__":
    main()

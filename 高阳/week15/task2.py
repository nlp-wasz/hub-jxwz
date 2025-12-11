import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "qwen-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-c4395731abd4446b8642c7734c8dbf56"
DEFAULT_OUTPUT_DIR = "mineru_output"


def load_segments(path: Path) -> List[str]:
    """Load text segments from mineru's content list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: List[str] = []
    for item in data:
        if item.get("type") in {"text", "discarded"}:
            text = str(item.get("text", "")).replace("\n", " ").strip()
            if text:
                segments.append(text)
    return segments


def build_index(segments: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create a simple TF-IDF index over character n-grams."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(segments)
    return vectorizer, matrix


def retrieve(
    query: str,
    segments: List[str],
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[float, str]]:
    """Return top-k segments with cosine similarity scores."""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix)[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), segments[i]) for i in order]


DEFAULT_QUESTIONS = [
    "java命名风格？",
    "如何进行并发处理？",
    "异常处理需要注意什么？",
]


def build_prompt(query: str, contexts: List[str]) -> str:
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "你是一个简洁的中文助理，基于给定参考段落回答问题。"
        "请用3-5句话概括答案，引用最相关的内容。\n\n"
        f"问题：{query}\n\n参考段落：\n{numbered}"
    )


def generate_with_qwen(
    query: str,
    hits: List[Tuple[float, str]],
    model: str,
    api_key: str,
    base_url: str,
) -> Optional[str]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[warn] : {exc}")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    contexts = [text for _, text in hits]
    prompt = build_prompt(query, contexts)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def ensure_mineru_output(pdf_path: Path, output_dir: Path) -> Path:
    content_path = output_dir /f"test.json"
    if content_path.exists():
        return content_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mineru",
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
    ]
    print(f"[info] Running mineru: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"mineru failed (exit {res.returncode}). stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    if not content_path.exists():
        raise FileNotFoundError(
            f"Expected mineru output missing: {content_path}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return content_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG over mineru output.")
    parser.add_argument(
        "--content",
        default="mineru_output/test.json",
        help="Path to mineru content list JSON.",
    )
    parser.add_argument(
        "--pdf",
        default="java.pdf",
        help="PDF path; if content JSON missing or --force-parse, run mineru.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="mineru output directory when parsing PDF.",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Force re-run mineru parsing even if content JSON exists.",
    )
    parser.add_argument("--question", help="Ask a custom question.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k contexts to retrieve.")
    args = parser.parse_args()

    if args.force_parse or not Path(args.content).exists():
        content_path = ensure_mineru_output(Path(args.pdf), Path(args.output_dir))
    else:
        content_path = Path(args.content)

    segments = load_segments(content_path)
    vectorizer, matrix = build_index(segments)

    if args.question:
        hits = retrieve(args.question, segments, vectorizer, matrix, top_k=args.topk)
        print(f"Q: {args.question}")
        for score, text in hits:
            print(f"- score={score:.3f} | {text}")
        answer = generate_with_qwen(
            args.question,
            hits,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        if answer:
            print("\n[LLM Answer]")
            print(answer)
    else:
        for q in DEFAULT_QUESTIONS:
            hits = retrieve(q, segments, vectorizer, matrix, top_k=args.topk)
            print(f"\nQ: {q}")
            answer = generate_with_qwen(
                q,
                hits,
                model=MODEL_NAME,
                api_key=API_KEY,
                base_url=BASE_URL,
            )
            if answer:
                print(f"\n[LLM Answer]\n{answer}\n")
            for score, text in hits:
                print(f"- score={score:.3f} | {text}")


if __name__ == "__main__":
    main()

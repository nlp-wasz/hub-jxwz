#!/usr/bin/env python
"""
命令行助手：通过简单的 RAG 步骤挑选最相关的 MCP 数学工具并执行。
会扫描 ../mcp_tools 下的工具，基于描述构建 TF-IDF 索引，对用户问题检索最相近的公式，
再调用排名最高（或 top-k）的工具，可接受用户参数。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dashscope import Generation  # type: ignore
except ImportError:  # noqa: BLE001 - 缺失时提示，允许无 LLM 回退
    Generation = None


@dataclass
class ToolRecord:
    name: str
    description: str
    tool: Any
    module_path: Path
    mcp_description: str = ""


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def load_mcp_tools(tool_dir: Path) -> List[ToolRecord]:
    """导入 MCP 工具模块并收集元数据。"""
    records: List[ToolRecord] = []
    for path in sorted(tool_dir.glob("tool*.py")):
        try:
            module = _load_module(path)
            mcp_server = getattr(module, "mcp", None)
            if mcp_server is None:
                continue
            tools = await mcp_server.get_tools()
            for name, tool in tools.items():
                desc = (tool.description or "").strip()
                if not desc:
                    desc = (getattr(getattr(tool, "fn", None), "__doc__", "") or "").strip()
                records.append(
                    ToolRecord(
                        name=name,
                        description=desc or name,
                        tool=tool,
                        module_path=path,
                        mcp_description=str(getattr(tool, "description", "")) or desc or name,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - 最佳努力加载，失败仅提示
            print(f"[警告] 加载 {path.name} 失败: {exc}", file=sys.stderr)
    return records


class RagSelector:
    """基于 TF-IDF 与余弦相似度的轻量级 RAG 选择器。"""

    def __init__(self, records: Sequence[ToolRecord]):
        corpus = [rec.description for rec in records]
        # 字符级 n-gram 可避免中文分词，直接处理描述文本。
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=4096,
        )
        self.doc_matrix = self.vectorizer.fit_transform(corpus)
        self.records = list(records)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[ToolRecord, float]]:
        if not query.strip():
            raise ValueError("问题不能为空")
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix)[0]
        ranked = list(zip(self.records, scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def parse_params(raw: str | None) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: B904 - 附带上下文抛出
        raise SystemExit(f"参数必须是合法的 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("参数 JSON 必须是对象，键为工具名，值为参数字典")
    return parsed  # type: ignore[return-value]


def format_content(content: Iterable[Any]) -> str:
    parts: List[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(item))
    return " ".join(parts)


async def run_tools(
    hits: Sequence[Tuple[ToolRecord, float]],
    params: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record, score in hits:
        tool_args = params.get(record.name) or params.get("*") or {}
        res = await record.tool.run(tool_args)
        results.append(
            {
                "tool": record.name,
                "module": str(record.module_path),
                "similarity": round(score, 4),
                "description": record.description,
                "structured": res.structured_content,
                "text": format_content(res.content),
            }
        )
    return results


def summarize_with_qwen(
    question: str,
    results: List[Dict[str, Any]],
    model: str,
    api_key: str | None,
) -> str:
    """用 Qwen 对工具结果做汇总；无可用 API 时退化为拼接输出。"""
    if Generation is None:
        return "未安装 dashscope，直接返回工具输出：\n" + "\n".join(
            f"- {r['tool']}: {r['text']}" for r in results
        )

    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        return "未提供 DASHSCOPE_API_KEY，直接返回工具输出：\n" + "\n".join(
            f"- {r['tool']}: {r['text']}" for r in results
        )

    context_lines: List[str] = []
    for idx, r in enumerate(results, 1):
        context_lines.append(
            f"工具{idx}: {r['tool']} (相似度 {r['similarity']})\n描述: {r['description']}\n结构化输出: {json.dumps(r['structured'], ensure_ascii=False, default=str)}\n文本输出: {r['text']}"
        )
    context = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": "你是一个回答公式问题的助手，基于给定的工具结果，用中文给出简洁的答案，如需要可以解释计算过程。",
        },
        {
            "role": "user",
            "content": f"用户问题: {question}\n下面是相关工具的输出，请综合给出回答：\n{context}",
        },
    ]

    try:
        resp = Generation.call(
            api_key=key,
            model=model,
            messages=messages,
            max_output_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001 - 保证 CLI 不因网络报错退出
        return f"调用 Qwen 失败，返回工具输出：{exc}\n" + "\n".join(
            f"- {r['tool']}: {r['text']}" for r in results
        )

    if getattr(resp, "status_code", None) not in (200, "200"):
        return f"Qwen 接口异常: {getattr(resp, 'message', resp)}\n" + "\n".join(
            f"- {r['tool']}: {r['text']}" for r in results
        )

    try:
        return resp.output["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - 容错解析
        return f"Qwen 响应解析失败，原始响应: {resp}"


async def async_main(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    default_tool_dir = base_dir.parent / "mcp_tools"
    tool_dir = Path(args.tool_dir).resolve() if args.tool_dir else default_tool_dir

    records = await load_mcp_tools(tool_dir)
    if not records:
        raise SystemExit(f"在 {tool_dir} 下未发现工具")

    selector = RagSelector(records)
    ranked = selector.search(args.question, top_k=max(1, args.topk))
    whitelist = ranked[: min(len(ranked), max(1, args.topk))]

    print("=== RAG 白名单（候选） ===")
    for idx, (rec, score) in enumerate(whitelist, start=1):
        print(f"{idx}. {rec.name}  相似度={score:.4f}  来源={rec.module_path.name}")
    print()

    to_execute = whitelist if args.run_all else whitelist[:1]
    if not to_execute:
        raise SystemExit("无可执行工具")

    params = parse_params(args.params)
    results = await run_tools(to_execute, params)

    print("=== 工具执行结果 ===")
    for item in results:
        print(f"[{item['tool']}] 源自 {item['module']} (相似度={item['similarity']})")
        print(f"结构化输出: {json.dumps(item['structured'], ensure_ascii=False, default=str)}")
        print(f"文本输出: {item['text']}")
        print("-" * 60)

    answer = summarize_with_qwen(
        question=args.question,
        results=results,
        model=args.model,
        api_key=args.api_key,
    )
    print("=== 汇总回答 ===")
    print(answer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 RAG 的 MCP 工具问答助手（调用 Qwen 汇总回答）")
    parser.add_argument("--question", required=True, help="用户问题，用于相似度检索")
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="白名单保留的工具数量（默认 3）",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="执行白名单内全部工具（默认只执行最优的一个）",
    )
    parser.add_argument(
        "--params",
        help="JSON 对象：工具名 -> 参数字典；'*' 作为通配符应用于所有候选",
    )
    parser.add_argument(
        "--tool-dir",
        help="自定义 mcp_tools 目录路径（默认 ../mcp_tools）",
    )
    parser.add_argument(
        "--model",
        default="qwen-plus",
        help="用于汇总回答的 Qwen 模型名称（默认 qwen-plus）",
    )
    parser.add_argument(
        "--api-key",
        help="DashScope API Key；若不提供则读取环境变量 DASHSCOPE_API_KEY",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

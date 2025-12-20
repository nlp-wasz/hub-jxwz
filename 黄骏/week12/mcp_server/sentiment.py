from typing import Annotated, Dict, List
import re

from fastmcp import FastMCP

mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="This server contains simple lexical sentiment analysis utilities.",
)

POSITIVE_TERMS = {
    "happy",
    "great",
    "excellent",
    "good",
    "awesome",
    "love",
    "like",
    "satisfied",
    "amazing",
    "nice",
    "delight",
    "enjoy",
    "success",
    "fantastic",
    "positive",
    "鼓舞",
    "开心",
    "满意",
    "喜欢",
    "棒",
    "赞",
    "高兴",
    "愉快",
    "顺利",
}

NEGATIVE_TERMS = {
    "sad",
    "bad",
    "terrible",
    "awful",
    "hate",
    "angry",
    "upset",
    "disappointed",
    "confused",
    "frustrated",
    "nervous",
    "poor",
    "worse",
    "negative",
    "depressing",
    "糟糕",
    "难过",
    "烦",
    "失望",
    "讨厌",
    "崩溃",
    "生气",
    "痛苦",
    "担忧",
    "压力",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\\w\\u4e00-\\u9fff]+", text.lower())


def _classify(score: int) -> str:
    if score > 1:
        return "positive"
    if score < -1:
        return "negative"
    return "neutral"


@mcp.tool
def analyze_text_sentiment(
    text: Annotated[str, "待分析的文本，将根据语气返回情感分类"]
) -> Dict[str, object]:
    """Analyzes the sentiment of the provided text and returns the label, score and matched keywords."""
    tokens = _tokenize(text)
    pos_hits = [token for token in tokens if token in POSITIVE_TERMS]
    neg_hits = [token for token in tokens if token in NEGATIVE_TERMS]
    score = len(pos_hits) - len(neg_hits)
    label = _classify(score)
    explanation_parts = []
    if pos_hits:
        explanation_parts.append(f"积极词汇: {', '.join(pos_hits[:5])}")
    if neg_hits:
        explanation_parts.append(f"消极词汇: {', '.join(neg_hits[:5])}")
    if not explanation_parts:
        explanation_parts.append("未匹配到典型情感词，根据整体语气判定为中性。")

    return {
        "text": text,
        "label": label,
        "score": score,
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "explanation": "；".join(explanation_parts),
    }


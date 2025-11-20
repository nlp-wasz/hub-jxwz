from typing import Annotated, Union
from textblob import TextBlob
import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-Emotion-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        if polarity > 0.1:
            sentiment = "积极"
            confidence = min(polarity * 2, 1.0)
        elif polarity < -0.1:
            sentiment = "消极"
            confidence = min(-polarity * 2, 1.0)
        else:
            sentiment = "中性"
            confidence = 1.0 - abs(polarity)

        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "confidence": round(confidence, 3),
            "text_length": len(text)
        }
    except Exception as e:
        return {
            "error": f"分析失败: {str(e)}",
            "sentiment": "未知"
        }


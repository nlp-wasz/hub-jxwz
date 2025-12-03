from typing import Annotated, Dict, Any
import re
from fastmcp import FastMCP

mcp = FastMCP(
    name="Sentiment-Analysis-MCP-Server",
    instructions="""This server contains tools for text sentiment analysis.""",
)

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]) -> Dict[str, Any]:
    """Classifies the sentiment of a given text into positive, negative, or neutral categories.
    
    Args:
        text: The text to analyze for sentiment
        
    Returns:
        A dictionary containing:
        - sentiment: The sentiment classification ('positive', 'negative', or 'neutral')
        - confidence: A confidence score between 0 and 1
        - details: Additional information about the analysis
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "details": "Empty text provided"
        }
    
    # 简单的基于关键词的情感分析
    # 在实际应用中，这里可以替换为更复杂的NLP模型或API调用
    
    # 积极情感关键词
    positive_words = [
        '好', '棒', '优秀', '喜欢', '爱', '开心', '快乐', '满意', '赞', '完美',
        'good', 'great', 'excellent', 'love', 'like', 'happy', 'wonderful', 'amazing', 'perfect'
    ]
    
    # 消极情感关键词
    negative_words = [
        '差', '糟糕', '讨厌', '恨', '难过', '失望', '生气', '愤怒', '垃圾', '无聊',
        'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'disappointed', 'angry', 'boring', 'garbage'
    ]
    
    # 转换为小写以便匹配
    text_lower = text.lower()
    
    # 计算积极和消极词汇的数量
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # 计算情感得分
    total_words = len(re.findall(r'\b\w+\b', text_lower))
    
    if total_words == 0:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "details": "No valid words found in text"
        }
    
    # 计算情感强度
    positive_ratio = positive_count / total_words
    negative_ratio = negative_count / total_words
    
    # 确定情感类别和置信度
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, 0.5 + positive_ratio)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, 0.5 + negative_ratio)
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    # 添加一些额外的分析信息
    details = {
        "positive_words_found": positive_count,
        "negative_words_found": negative_count,
        "total_words_analyzed": total_words
    }
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "details": details
    }
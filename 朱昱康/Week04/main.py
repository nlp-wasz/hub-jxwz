#!/usr/bin/env python3
"""
外卖评论文本分类服务
启动命令：python main.py
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from model import TextClassificationModel

# 初始化FastAPI应用
app = FastAPI(
    title="外卖评论分类",
    description="基于BERT的中文文本情感分析",
    version="1.0.0"
)

# 初始化模型
print("正在加载模型...")
model = TextClassificationModel()
print("模型加载完成！")

@app.get("/")
async def root():
    """根路径信息"""
    return {
        "message": "外卖评论文本分类服务已启动",
        "usage": "/classify?text=你的评论文本"
    }

@app.get("/classify")
async def classify_text(text: str):
    """
    文本分类接口
    
    Args:
        text: 要分类的中文评论文本
        
    Returns:
        包含预测类别和置信度的响应
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    if len(text.strip()) > 512:
        raise HTTPException(status_code=400, detail="文本长度不能超过512字符")
    
    try:
        result = model.predict(text.strip())
        return {
            "text": text.strip(),
            "sentiment": "正面" if result["predicted_class"] == 1 else "负面",
            "confidence": round(result["confidence"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

if __name__ == "__main__":
    print("正在启动服务...")
    print("访问地址: http://localhost:8000")
    print("使用示例: http://localhost:8000/classify?text=很好吃")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
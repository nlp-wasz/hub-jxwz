"""
外卖评价情感分类API服务
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import time
import traceback
import uvicorn
from waimai_bert_model import predict_sentiment

# 创建FastAPI应用
app = FastAPI(
    title="外卖评价情感分类API",
    description="基于BERT的外卖评价情感分析服务",
    version="1.0.0"
)


# 请求数据模型
class WaimaiClassifyRequest(BaseModel):
    """外卖评价分类请求"""
    request_id: Optional[str] = Field(None, description="请求ID，用于追踪")
    text: Union[str, List[str]] = Field(..., description="待分类的文本，支持单条或多条")

    class Config:
        schema_extra = {
            "example": {
                "request_id": "test_001",
                "text": "菜品很好吃，送餐也很快，五星好评！"
            }
        }


class SinglePrediction(BaseModel):
    """单条预测结果"""
    text: str = Field(..., description="原始文本")
    label: str = Field(..., description="预测标签：正面/负面")
    label_id: int = Field(..., description="标签ID：0-负面，1-正面")
    confidence: float = Field(..., description="预测置信度")
    probabilities: dict = Field(..., description="各类别概率")


class WaimaiClassifyResponse(BaseModel):
    """外卖评价分类响应"""
    request_id: Optional[str] = Field(None, description="请求ID")
    success: bool = Field(..., description="请求是否成功")
    results: Union[SinglePrediction, List[SinglePrediction]] = Field(..., description="预测结果")
    classify_time: float = Field(..., description="分类耗时（秒）")
    error_msg: str = Field("", description="错误信息")

    class Config:
        schema_extra = {
            "example": {
                "request_id": "test_001",
                "success": True,
                "results": {
                    "text": "菜品很好吃，送餐也很快，五星好评！",
                    "label": "正面",
                    "label_id": 1,
                    "confidence": 0.9876,
                    "probabilities": {
                        "负面": 0.0124,
                        "正面": 0.9876
                    }
                },
                "classify_time": 0.123,
                "error_msg": ""
            }
        }


@app.post("/classify", response_model=WaimaiClassifyResponse)
async def classify_sentiment(request: WaimaiClassifyRequest):
    """
    外卖评价情感分类接口
    
    Args:
        request: 分类请求，包含待分类的文本
        
    Returns:
        WaimaiClassifyResponse: 分类结果
    """
    start_time = time.time()

    # 初始化响应
    response = WaimaiClassifyResponse(
        request_id=request.request_id,
        success=False,
        results=[],
        classify_time=0.0,
        error_msg=""
    )

    try:
        # 调用BERT模型进行预测
        prediction_results = predict_sentiment(request.text)

        # 处理结果格式
        if isinstance(prediction_results, dict):
            # 单条结果
            response.results = SinglePrediction(**prediction_results)
        else:
            # 多条结果
            response.results = [SinglePrediction(**result) for result in prediction_results]

        response.success = True
        response.error_msg = "ok"

    except Exception as e:
        response.success = False
        response.error_msg = f"分类失败: {str(e)}"
        response.results = []

        # 记录详细错误信息
        print(f"分类错误: {traceback.format_exc()}")

    finally:
        response.classify_time = round(time.time() - start_time, 4)

    return response


@app.post("/batch_classify")
async def batch_classify_sentiment(texts: List[str], request_id: Optional[str] = None):
    """
    批量情感分类接口
    
    Args:
        texts: 待分类的文本列表
        request_id: 可选的请求ID
        
    Returns:
        批量分类结果
    """
    request = WaimaiClassifyRequest(
        request_id=request_id,
        text=texts
    )
    return await classify_sentiment(request)


def start_server(host="0.0.0.0", port=8000, reload=False):
    """
    启动FastAPI服务器
    
    Args:
        host: 服务器主机地址
        port: 服务器端口
        reload: 是否启用热重载
    """
    print("启动外卖评价情感分类API服务！")
    print("API服务: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload
    )


def get_app():
    """
    获取FastAPI应用实例
    
    Returns:
        FastAPI: 应用实例
    """
    return app


# 启动配置
if __name__ == "__main__":
    start_server()

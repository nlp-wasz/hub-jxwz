import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.evaluations import router as evaluation_router
from logger import logger

app = FastAPI(
    title="外卖评价分类系统",
    description="基于BERT的外卖评价文本分类系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 注册路由
app.include_router(evaluation_router, prefix="/api/v1", tags=["评价分类"])

@app.get("/")
async def root():
    """
    首页
    """
    return {"message": "欢迎使用外卖评价分类系统"}

@app.get("/health")
async def health():
    """
    健康检查
    """
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("外卖评价分类系统启动...")
    
    # 内部预热模型
    logger.info("预热模型...")
    try:
        from core.model.bert import classify_for_bert
        # 直接调用模型函数进行预热
        classify_for_bert("这家外卖很好吃，包装也很好")
        logger.info("模型预热完成")
    except Exception as e:
        logger.error(f"模型预热失败: {str(e)}")
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)

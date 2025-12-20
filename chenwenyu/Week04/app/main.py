import os
from pathlib import Path
import uvicorn
import signal
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List
from app.schemas import PredictionRequest, PredictionResult, HealthResponse, TrainRequest
from app.predictor import load_predictor
from app import bert_model  # 导入你的训练模块

app = FastAPI(
    title="BERT文本分类API",
    description="基于微调BERT模型的文本分类服务",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Helper Utilities
def safe_convert(obj: Any) -> Any:
    """安全转换任何对象为JSON可序列化的格式"""
    if hasattr(obj, 'item'):
        # 处理NumPy标量
        return obj.item()
    elif hasattr(obj, 'tolist'):
        # 处理NumPy数组
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert(item) for item in obj]
    else:
        return obj

def safe_serialize(obj: Any) -> Any:
    """安全序列化对象"""
    try:
        # 使用NumPy的item()方法
        if hasattr(obj, 'item'):
            return obj.item()
        # 使用NumPy的tolist()方法
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [safe_serialize(item) for item in obj]
        else:
            return obj
    except:
        return str(obj)
     
# 在应用程序实例上存储状态
app.state.predictor = None

@app.on_event("startup")
async def startup_event():
    """启动时加载最新模型"""
    try:
        # 加载最新训练的模型
        model_dir="./app/models/bert-finetuned-epoch1"
        model_path=Path(model_dir)
        print(f"尝试加载模型从: {model_path}")
        print(f"路径是否存在: {model_path.exists()}")

        if model_path.exists():
            predictor_instance = load_predictor(str(model_dir))
            app.state.predictor = predictor_instance  # 存储到应用状态
            print("✅ Model loaded successfully")
            print(f"Predictor type: {type(predictor_instance)}")
            if hasattr(predictor_instance, 'device'):
                print(f"Predictor device: {predictor_instance.device}")
        else:
            print(f"❌ 模型路径不存在: {model_dir}")
            # 列出目录内容以便调试
            models_dir = "./app/models"
            models_path=Path(models_dir)
            if models_path.exists():
                print(f"models目录内容: {list(models_path.iterdir())}")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        import traceback
        traceback.print_exc()

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "BERT文本分类服务已启动"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    predictor = app.state.predictor
    print(f"Health check - predictor: {predictor}")  # 调试信息

    return {
        "status": "healthy", 
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "none"
    }

# 添加一个调试端点来检查应用状态
@app.get("/debug")
async def debug():
    return {
        "predictor_exists": app.state.predictor is not None,
        "predictor_type": str(type(app.state.predictor)),
        "app_state_keys": list(app.state.__dict__.keys())
    }

@app.get("/debug-predict")
async def debug_predict():
    """调试预测结果的格式"""
    test_text = "这个产品很好用"
    
    if not app.state.predictor:
        return {"error": "Model not loaded"}
    
    try:
        result = app.state.predictor.predict(test_text)
        
        # 分析结果类型
        result_info = {
            "type": str(type(result)),
            "value": str(result),
            "is_numpy": hasattr(result, 'dtype') or hasattr(result, 'item'),
        }
        
        # 尝试转换
        try:
            converted = safe_convert(result)
            result_info["converted"] = converted
            result_info["converted_type"] = str(type(converted))
        except Exception as conv_e:
            result_info["conversion_error"] = str(conv_e)
        
        return result_info
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/predict", response_model=List[PredictionResult])
async def predict(request: PredictionRequest):
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts cannot be empty")
    
    try:
        #print("predict batch:", request.texts)
        results = predictor.predict(request.texts)
        # 安全转换NumPy类型
        serialized_results = safe_serialize(results)
        #print(f"type serialized_result:{type(serialized_results)}, content: {serialized_results}")
        if isinstance(serialized_results, list) and len(serialized_results) > 0:
            resp=[]
            for result in serialized_results:
                resp.append({
                "text": result.get("text"),
                "predicted_label": str(result.get("predicted_label")),
                "predicted_class": result.get("predicted_class"),
                "confidence": round(result.get("confidence", 0), 4),
            })
            return resp
        else:
            return [{"error": "No prediction result"}]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{text}")
async def predict_single(text: str):
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predictor.predict_single(text)
        # 安全转换NumPy类型
        serialized_result = safe_serialize(result)
        #print("type serialized_result:",type(serialized_result))
        if isinstance(serialized_result, list) and len(serialized_result) > 0:
            result = serialized_result[0]
            # 只返回关键信息
            return {
                "text": text,
                "predicted_label": str(result.get("predicted_label")),
                "predicted_class": result.get("predicted_class"),
                "confidence": round(result.get("confidence", 0), 4),
            }
        else:
            return {"error": "No prediction result"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shutdown")
async def shutdown():
    """关闭FASTAPI服务器"""
    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "服务器正在关闭..."}

if False:
    @app.post("/train")
    async def train_model(request: TrainRequest):
        """触发模型训练"""
        try:
            # 调用你的训练函数
            bert_model.main_train(epochs=request.epochs, batch_size=request.batch_size)
            return {"status": "training_started", "epochs": request.epochs}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
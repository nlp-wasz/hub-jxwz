  from fastapi import FastAPI
from pydantic import BaseModel

from Week04.homework.predict_bert import BertTextClassifier

app = FastAPI(title="BERT文本分类API", version="1.0")

# 加载模型
classifier = BertTextClassifier(
    '../homework/bert/best_model',
    '../homework/bert/label_encoder.pkl'
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "BERT文本分类API，请使用POST请求访问/classify端点"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    try:
        result = classifier.predict(request.text)
        return {
            "text": request.text,
            **result
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

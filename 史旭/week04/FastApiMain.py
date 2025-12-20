from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse

from DataType import Request, Response, RequestPredict
from BertTrainPred1 import self_bert_train_pred
from BertTrainPred2 import auto_bert_train_eval
import time
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 创建 fastapi 对象
fastapi = FastAPI(
    title='分类项目案例',
    description='分类项目案例',
    version='0.0.1',
    docs_url=None
)


# fastapi页面swagger方式默认加载境外资源，这里修改为国内资源
@fastapi.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API 文档",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


@fastapi.post("/self_bert_model_train", description="完成bert模型的训练，以及精度预测（手动定义训练评估逻辑）")
def self_bert_model_train1(request: Request) -> Response:
    # 记录 训练开始时间
    start_time = time.time()

    # 调用 bert模型 训练方法
    response = self_bert_train_pred(request)

    # 记录 训练结束时间
    end_time = time.time()

    # 计算 执行时间
    response.run_time = f"{end_time - start_time:.4f}s"

    return response


@fastapi.post("/self_text_predict", description="传递文本，预测标签（手动训练模型）")
def self_text_predict(request: RequestPredict) -> str:
    pred_text = request.new_text

    # 加载 训练好的模型
    bert_model = joblib.load("./data/bert_train_model.bin")

    # 分词器处理 预测文本信息
    tokenizer = AutoTokenizer.from_pretrained("../../../models/google-bert/bert-base-chinese")
    pred_encode = tokenizer([pred_text], max_length=32, truncation=True, padding="max_length", return_tensors="pt")

    # 预测
    outputs = bert_model(input_ids=pred_encode["input_ids"], attention_mask=pred_encode["attention_mask"])
    logits = outputs.logits
    max_index = logits.argmax(dim=-1).item()

    return f"{pred_text} -> 好评" if max_index == 1 else f"{pred_text} -> 差评"


@fastapi.post("/auto_bert_model_train", description="完成bert模型的训练，以及精度预测（Trainer自动训练评估模型）")
def auto_bert_model_train2(request: Request) -> str:
    # 记录 训练开始时间
    start_time = time.time()

    # 调用 bert模型 训练方法
    auto_bert_train_eval(request)

    # 记录 训练结束时间
    end_time = time.time()

    # 计算 执行时间
    run_time = f"{end_time - start_time:.4f}s"

    return run_time


@fastapi.post("/auto_text_predict", description="传递文本，预测标签（自动 训练模型）")
def auto_text_predict(request: RequestPredict) -> str:
    pred_text = request.new_text

    # 分词器 处理
    tokenizer = AutoTokenizer.from_pretrained("../../../models/google-bert/bert-base-chinese")
    pred_encode = tokenizer([pred_text], max_length=32, truncation=True, padding="max_length", return_tensors="pt")

    # 加载模型
    bert_model = AutoModelForSequenceClassification.from_pretrained("./data/autoTrain", num_labels=2)

    # 模型预测
    outputs = bert_model(input_ids=pred_encode["input_ids"], attention_mask=pred_encode["attention_mask"])
    logits = outputs.logits

    # 最大值索引位置
    max_index = logits.argmax(dim=-1).item()

    return f"{pred_text} -> 好评" if max_index == 1 else f"{pred_text} -> 差评"

from datetime import datetime
from fastapi import APIRouter
from logger import logger
from schema.evaluation import EvaluationClassifyRequest, EvaluationClassifyResponse
from core.model.bert import classify_for_bert

router = APIRouter()

@router.post("/evaluation_classify/bert")
async def bret_classify(request: EvaluationClassifyRequest):
    """
    使用BERT进行评价文本分类
    :param request:
    :return:
    """
    start_time = datetime.now()

    response = EvaluationClassifyResponse(
        request_id = request.request_id,
        request_evaluation = request.request_evaluation,
        classify_result = "",
        classify_time = 0,
        error_message = ""
    )

    try:
        response.classify_result = classify_for_bert(request.request_evaluation)
        response.error_message = "success"
    except Exception as e:
        response.classify_result = ""
        response.error_message = str(e)

    response.classify_time = (datetime.now() - start_time).total_seconds()

    return response
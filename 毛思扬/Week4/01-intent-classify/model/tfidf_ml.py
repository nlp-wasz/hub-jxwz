# file: /Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week4/01-intent-classify/model/tfidf_ml.py
from typing import Union, List
import jieba
import pandas as pd
from config import TFIDF_MODEL_PKL_PATH
from joblib import load
import numpy as np

# 加载模型
tfidf, model = load(TFIDF_MODEL_PKL_PATH)

# 停用词
cn_stopwords = pd.read_csv('http://mirror.coggle.club/stopwords/baidu_stopwords.txt', header=None)[0].values


def model_for_tfidf(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        query_words = " ".join([x for x in jieba.lcut(request_text) if x not in cn_stopwords])
        predict_result = model.predict(tfidf.transform([query_words]))[0]
        # 将 numpy.int64 转换为 Python 原生 int 类型
        classify_result = int(predict_result) if isinstance(predict_result, (np.int64, np.integer)) else predict_result
    elif isinstance(request_text, list):
        query_words = []
        for text in request_text:
            query_words.append(
                " ".join([x for x in jieba.lcut(text) if x not in cn_stopwords])
            )
        predict_result = model.predict(tfidf.transform(query_words))
        # 将所有 numpy 类型转换为 Python 原生类型
        classify_result = [int(x) if isinstance(x, (np.int64, np.integer)) else x for x in predict_result]
    else:
        raise Exception("格式不支持")

    return classify_result
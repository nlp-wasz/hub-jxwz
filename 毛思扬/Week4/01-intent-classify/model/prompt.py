from typing import Union, List

import numpy as np
import openai
import pandas as pd
from config import (
    LLM_OPENAI_API_KEY,
    LLM_OPENAI_SERVER_URL,
    LLM_MODEL_NAME,
    TFIDF_MODEL_PKL_PATH
)
from joblib import load

train_data = pd.read_csv('assets/dataset/dataset.csv', sep='\t', header=None)

tfidf, _ = load(TFIDF_MODEL_PKL_PATH)
train_tfidf = tfidf.transform(train_data[0])

client = openai.Client(base_url=LLM_OPENAI_SERVER_URL, api_key=LLM_OPENAI_API_KEY)

PROMPT_TEMPLATE = '''你是一个情感分类的专家，请结合待选类别和参考例子进行情感分类。
待选类别：好评 / 差评

历史参考例子如下：
差评："不错,就是菜有点时间长了,不新鲜了"
差评：菜品一般，服务态度也一般
差评：环境一般，菜品也一般
差评：谢谢，服务非常好，继续支持
差评：丽华饭量小了点

好评：特别好吃，量特大，而且送餐特别快，特别特别棒
好评：口感好的很，速度快！
好评：相当好吃的香锅，分量够足，味道也没的说。
好评：好吃！速度！包装也有品质，不出家门就能吃到餐厅的味道！
好评：量大味道好，送餐师傅都很好



待识别的文本为：{0}
只需要输出情感类别（从待选类别中选一个），不要其他输出。'''




def model_for_gpt(request_text: Union[str, List[str]]) -> List[str]:
    classify_result: Union[str, List[str]] = []

    if isinstance(request_text, str):
        tfidf_feat = tfidf.transform([request_text]) # 一个文本
        request_text = [request_text]
    elif isinstance(request_text, list):
        tfidf_feat = tfidf.transform(request_text) # 多个文本
    else:
        raise Exception("格式不支持")

    for query_text, idx in zip(request_text, range(tfidf_feat.shape[0])):
        # 动态提示词
        ids = np.dot(tfidf_feat[idx], train_tfidf.T) # 计算待推理的文本与训练哪些最相似
        top10_index = ids.toarray()[0].argsort()[::-1][:10]

        # 组织为字符串
        dynamic_top10 = ""
        for similar_row in train_data.iloc[top10_index].iterrows():
            dynamic_top10 += similar_row[1][0] + " -> " + similar_row[1][1].replace("-", "") + "\n"

        response = client.chat.completions.create(
            # 云端大模型、云端token
            # 本地大模型，本地大模型地址
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    query_text, dynamic_top10, "/".join(list(train_data[1].unique()))
                )},
            ],
            temperature=0,
            max_tokens=64,
        )

        classify_result.append(response.choices[0].message.content)

    return classify_result

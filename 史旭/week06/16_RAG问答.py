import json
import pdfplumber
import requests
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# RAG问答流程（多路召回 + 融合排序 + 重排序 + pormpt提示词构建 + 调用LLM（http 和 openai））

# 提前加载 重排序 分词器和模型
tokenizer = AutoTokenizer.from_pretrained("../models/BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("../models/BAAI/bge-reranker-base")
rerank_model.eval()


# 调用大模型方法（http）
def get_llm_qwen_http(prompt):
    # 调用地址
    model_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    # model_url = "http://localhost:11434/api/generate"

    # 请求头
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-04ab3d7290e243dda1badc5a1d5ac858"
    }

    # 请求体
    data = {
        # "model": "qwen3:0.6b",
        # "prompt": prompt,
        "model": "qwen-plus",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    # 调用大模型
    error_number = 0
    while True:
        if error_number >= 5:
            return "连接超时！"
        try:
            response = requests.post(model_url, headers=header, json=data)
            return response.json()
        except:
            error_number += 1


# 调用大模型方法（openai）
def get_llm_qwen_openai(prompt):
    qwen_model = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-04ab3d7290e243dda1badc5a1d5ac858"
    )

    # 调用大模型
    error_number = 0
    while True:
        if error_number >= 5:
            return "连接超时！"
        try:
            response = qwen_model.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            return response
        except:
            error_number += 1


# 1.读取 questions.json 问题信息
questions = json.load(open("./data/questions.json", "r", encoding="utf-8"))

# 2.读取 pdf 每页文本内容
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
pages = pdf.pages

# 记录 每一页的文本内容
pages_content = {}
for page_i, page in enumerate(pages):
    # 获取 每页 文本内容
    text = page.extract_text()

    # 记录保存
    pages_content[f"page_{page_i + 1}"] = text

# 3.多路召回
bm25 = json.load(open("./data/02_BM25_question_top10.json", "r", encoding="utf-8"))
bge = json.load(open("./data/03_bge_question_top10.json", "r", encoding="utf-8"))

# 循环每个问题，进行多路召回结果的 融合排序
for index, (bm25_q, bge_q) in enumerate(zip(bm25[:20], bge[:20])):
    # 获取 bm25_q 和 bge_q 对于每个问题检索的结果
    bm25_top10 = bm25_q["reference"]
    bge_top10 = bge_q["reference"]

    # 融合 打分
    fusion_score = {}
    k = 60
    for bm25_i, reference in enumerate(bm25_top10):
        # reference ->  "page_115"
        if reference not in fusion_score:
            fusion_score[reference] = 1 / (k + bm25_i)
        else:
            fusion_score[reference] += 1 / (k + bm25_i)

    for bge_i, reference in enumerate(bge_top10):
        # reference ->  "page_115"
        if reference not in fusion_score:
            fusion_score[reference] = 1 / (k + bge_i)
        else:
            fusion_score[reference] += 1 / (k + bge_i)

    # 对融合打分后的结果 进行排序
    fusion_sorted = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    # 对融合排序后的结果 进行重排序（使用重排序模型）
    # 先构建句子对（即 问题 -> 页码内容），通过句子对让模型重新打分
    fusion_rerank = []
    for sorted_i, (page_index, page_score) in enumerate(fusion_sorted[:2]):
        # page_index -> "page_115"

        # 根据 提取出来的页码 page_index，构建句子对
        fusion_rerank.append([
            bm25_q["question"],  # 问题
            pages_content[page_index],  # 对应页码 内容
        ])

    # 对 top3 句子对进行分词处理
    fusion_tokenizer = tokenizer(fusion_rerank, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # fusion_tokenizer 包含三部分：input_ids  attention_mask  token_type_ids
    # 将其 转换为 字典类型，作为输入传递给 重排序模型（基于BERT模型）
    fusion_inputs = {key: value_tensor for key, value_tensor in fusion_tokenizer.items()}

    # 重排序模型 重新打分（不改变向量位置，只重新打分，不会进行排序操作）
    with torch.no_grad():
        fusion_rerank_logits = rerank_model(**fusion_inputs).logits

        # fusion_rerank_logits -> (topN, 1)：每个向量 对应的 打分结果
        # 获取 重排序 后的 top1
        max_score_index = fusion_rerank_logits.view(-1).argmax(-1).item()

    # max_score_index 对应的是 融合排序后的结果，并非页码
    # 获取 对应的 页码
    max_score_page_number = fusion_sorted[0][0]

    # 提取 问题信息，问题对应页码内容  构建prompt
    curr_question = bm25_q["question"]
    curr_page_content = pages_content[max_score_page_number]

    # 构建 prompt 提示词工程
    prompt = '''你是一个专业的汽车领域专家，请根据下面提供的资料，回答用户提出的问题。
    回答内容必须是提供资料中的内容，如果用户提出的问题与汽车无关，请返回“无法回答”，
    如果用户提出的问题不在资料范围内，请返回“无法回答”，如果提供的资料可以回答用户的问题，请根据资料回答用户问题。
    
    资料：{0}
    
    问题：{1}
    
    '''.format(curr_page_content, curr_question)

    # 调用 LLM
    print(f"问题：{curr_question}")
    # 获取 回答结果（http 方式）
    # qwen_response = get_llm_qwen_http(prompt)
    # answer = qwen_response["choices"][0]["message"]["content"]

    # 获取 回答结果（openai 方式）
    qwen_response = get_llm_qwen_openai(prompt)
    answer = qwen_response.choices[0].message.content

    # 将 answer 和 页码max_score_page_number 保存在questions中
    questions[index]["answer"] = answer
    questions[index]["reference"] = max_score_page_number

# 保存 questions 对象
with open("./data/07_rag_llm_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

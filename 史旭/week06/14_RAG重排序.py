import pdfplumber
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 通过 多路召回 对召回结果进行融合排序，之后再通过重排序模型（进行重排序），最终得到topN
# 加载 重排序模型（后续使用）-- 分词器和重排序模型
tokenizer = AutoTokenizer.from_pretrained("../models/BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("../models/BAAI/bge-reranker-base")
# 改为 评估 模式
rerank_model.eval()

# 1.读取 question.json
questions = json.load(open("./data/questions.json", "r", encoding="utf-8"))

# 2.读取 pdf
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
pages = pdf.pages

pages_content = []
for page_i, page in enumerate(pages):
    # 获取 每页内容
    text = page.extract_text()
    pages_content.append({
        "page": page_i + 1,
        "content": text
    })

# 3.加载 多路检索 结果信息
bm25 = json.load(open("./data/02_BM25_question_top10.json", "r", encoding="utf-8"))
bge = json.load(open("./data/03_bge_question_top10.json", "r", encoding="utf-8"))

# 4.对每个问题 进行多路召回（融合排序）
for question_i, (bm25_q, bge_q) in enumerate(zip(bm25, bge)):
    print(f"问题编号：{question_i + 1}, 问题标题：{bm25_q["question"]}")
    # ①获取 bm25 和 bge 分别对每个问题 检索的结果
    bm25_top10 = bm25_q["reference"]
    bge_top10 = bge_q["reference"]

    # ②融合排序（对每个问题的结果 进行统一打分，RRF算法  k是一个超参数 证明得到60为最优）
    fusion_result = {}
    k = 60

    # 先对 bm25_top10 循环
    for bm25_i, reference in enumerate(bm25_top10):
        # reference -> page_116：将其作为key，记录到 fusion_result  其value为对应的分数
        if reference not in fusion_result:
            fusion_result[reference] = 1 / (k + bm25_i)
        else:
            fusion_result[reference] += 1 / (k + bm25_i)

    # 再对 bge_top10 循环（谁先谁后 无区别）
    for bge_i, reference in enumerate(bge_top10):
        # reference -> page_116：将其作为key，记录到 fusion_result  其value为对应的分数
        if reference not in fusion_result:
            fusion_result[reference] = 1 / (k + bge_i)
        else:
            fusion_result[reference] += 1 / (k + bge_i)

    # ③最终得到 多路召回后 融合排序的打分结果（自定义打分，证明得到这种打分方式在RAG流程中最优）
    # 对 fusion_result 进行排序（按照 value 值，即score分数排序）
    # fusion_result.items()返回元组，(key, value)
    fusion_sort = sorted(fusion_result.items(), key=lambda item: item[1], reverse=True)

    # ④对 融合排序结果 fusion_rerank 重排序（重排序模型）
    # 选取 融合排序后的 top5 进行重排序
    fusion_rerank = fusion_sort[:3]

    # 根据 fusion_rerank，获取对应的 (问题, pdf文本内容)，组成句子对（对句子对进行分词以及模型评估，获取得分）
    question_content = []
    for tuple_page in fusion_rerank:
        # tuple_page -> ("page_115", 0.225) 形式（str分割 获取页码）
        page_index = tuple_page[0].split("_")[1]

        # 构建句子对 -> (question, content)
        question_content.append([bm25_q["question"], pages_content[int(page_index) - 1]["content"]])

    # 通过 分词器 对 question_content 进行分词处理
    question_content_tokenizer = tokenizer(question_content, padding=True, truncation=True, max_length=512,
                                           return_tensors="pt")

    # question_content_tokenizer 由三部分组成，input_ids  attention_mask  token_type_ids
    # 将其转换为 dict，作为输入input 传递给model进行评估
    question_content_inputs = {key: input_tensor for key, input_tensor in question_content_tokenizer.items()}

    # 模型 评估（禁用 pytorch 梯度计算）
    with torch.no_grad():
        # 重排序 只对传入向量做打分操作，不会进行排序（因此不影响 fusion_rerank 的顺序）
        rerank_result = rerank_model(**question_content_inputs).logits

        # rerank_result -> (topN, 1) 评估向量对应的分数（展平为 一维向量）
        rerank_result = rerank_result.view(-1).float()

        # 获取 重排序 后最大分数 对应的索引位置
        max_score_page_index = rerank_result.argmax(-1).item()

        # 获取 多路召回 + 融合排序 + 重排序 后的top1
        page_number = fusion_sort[max_score_page_index][0]
        questions[question_i]["reference"] = page_number

# print(json.dumps(questions, indent=4, ensure_ascii=False))
with open("./data/05_rerank_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

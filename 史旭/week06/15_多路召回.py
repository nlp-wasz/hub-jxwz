import json

# 将 多路检索 结果进行融合排序（即 多路召回）
# 1.读取 question.json 信息
questions = json.load(open('./data/questions.json', "r", encoding="utf-8"))

# 1.读取 bm25 和 bge 两种方式检索结果文件（json）
bm25 = json.load(open("./data/02_BM25_question_top10.json", "r", encoding="utf-8"))
bge = json.load(open("./data/03_bge_question_top10.json", "r", encoding="utf-8"))

# 多路召回逻辑（循环不同的检索结果，根据问题所在页码，对其进行融合打分）
# 超参数k：60为最优
k = 60
for index, (bm25_q, bge_q) in enumerate(zip(bm25, bge)):
    # 获取 bm25_q 和 bge_q 问题对应的检索结果
    bm25_reference = bm25_q["reference"]
    bge_reference = bge_q["reference"]

    # 对两种检索结果 分别循环
    fusion_reference = {}
    for bm25_i, reference, in enumerate(bm25_reference):
        # reference ->  "page_115"  这种形式
        # 打分
        if reference not in fusion_reference:
            fusion_reference[reference] = 1 / (k + bm25_i)
        else:
            fusion_reference[reference] += 1 / (k + bm25_i)

    for bge_i, reference in enumerate(bge_reference):
        # reference ->  "page_115"  这种形式
        # 打分
        if reference not in fusion_reference:
            fusion_reference[reference] = 1 / (k + bge_i)
        else:
            fusion_reference[reference] += 1 / (k + bge_i)

    # 对 多路召回 融合打分后的 结果 进行排序
    fusion_sorted = sorted(fusion_reference.items(), key=lambda item: item[1], reverse=True)

    # 最后重排序，不属于多路召回（多路召回：融合排序）

    # 保存 融合排序后的结果
    questions[index]["reference"] = fusion_sorted

# 保存 questions 结果到 json 文件
json.dump(questions, open("./data/06_recall_questions.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

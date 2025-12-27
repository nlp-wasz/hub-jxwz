import jieba
import pdfplumber
import json
from rank_bm25 import BM25Okapi

# 与 TFIDF类似，都是基于词频向量，计算相似性得分（但是 BM25 效果更好）
# 因此在RAG流程中 采用 BM25全文检索 + SBert语义向量检索 多路检索

# 1.读取 question问题信息
with open("./data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 2.提取 pdf 每页文本内容
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
pages = pdf.pages

pages_content = []
for index, page in enumerate(pages):
    # 读取文本内容
    text = page.extract_text()
    pages_content.append({
        "page": index + 1,
        "content": text
    })

# 3.分词处理
questions_jieba = [list(jieba.cut(question["question"])) for question in questions]
pages_content_jieba = [list(jieba.cut(page_content["content"])) for page_content in pages_content]

# 3.BM25 只需要根据 pdf中的文本分词内容 构建词汇表（无需与questions一同构建）
# 创建 BM25 对象
bm25 = BM25Okapi(pages_content_jieba)

# 4.循环 questions_jieba，通过bm25对象 获取相似性得分 top1
for index, question_jiebe in enumerate(questions_jieba):
    # bm25.get_scores()：计算得分
    score = bm25.get_scores(question_jiebe)

    # 获取 top1
    max_score_page_index = score.argsort(axis=-1).reshape(-1)[::-1][0]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = f"page_{max_score_page_index + 1}"

# 将 questions 对象，保存至 文件 中
with open("./data/02_BM25_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# 5.循环 questions_jieba，通过bm25对象 获取相似性得分 top10
for index, question_jiebe in enumerate(questions_jieba):
    # bm25.get_scores()：计算得分
    score = bm25.get_scores(question_jiebe)

    # 获取 top1
    max_score_pages_index = score.argsort(axis=-1).reshape(-1)[::-1][:10]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = [f"page_{page_index + 1}" for page_index in max_score_pages_index]

# 将 questions 对象，保存至 文件 中
with open("./data/02_BM25_question_top10.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

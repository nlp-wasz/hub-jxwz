import json
import pdfplumber
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# 在RAG流程中，通常采用 全文检索+向量检索 多路检索模式
# 全文检索：TFIDF 或者  BM25  都是词频检索（通过倒排索引，根据问题的关键词，快速定位文档位置）
# 可以做精确检索，但对语义不敏感

# TFIDF 检索
# 1.首先读取 问题集合（从json文件中读取）
with open("./data/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# 2.读取pdf文件，获取每一页文本内容
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
pages = pdf.pages

pages_content = []
for index, page in enumerate(pages):
    # 读取每一页内容
    text = page.extract_text()
    # 将每一页 页码和内容 存储到page_content中
    pages_content.append({
        "page": index + 1,
        "content": text
    })

# 3.对 questions 和 pages_content 进行分词处理（用于构建词频表 和 获取词频向量）
questions_jieba = [" ".join(jieba.cut(question["question"])) for question in questions]
pages_content_jieba = [" ".join(jieba.cut(page_content["content"])) for page_content in pages_content]

# 4.TFIDF构建词汇表，获取词频向量（对 问题和pdf内容 统一构建词汇表）
tfidf = TfidfVectorizer()
tfidf.fit(questions_jieba + pages_content_jieba)

# 获取词频向量
questions_vector = tfidf.transform(questions_jieba)
pages_content_vector = tfidf.transform(pages_content_jieba)

# 5.分词后 进行归一化处理（使得 计算点积时，其结果就是 余弦相似度）
# 也可以在后面直接使用 sklearn.metrics.pairwise.cosine_similarity 计算余弦相似度
questions_normalize = normalize(questions_vector)
pages_content_normalize = normalize(pages_content_vector)

# 6.循环每一个 归一化 后的问题向量，计算其与所有page向量的点积（余弦相似度），获取最相似的topN
for index, question_normalize in enumerate(questions_normalize):
    # 计算 点积
    # score
    score = question_normalize @ pages_content_normalize.T

    # score 转换为 numpy.ndarray 类型  ->  (1, pages_size)
    score_numpy = score.toarray()

    # argsort() 排序，并通过索引方式 获取 TOP1
    max_score_page_index = score_numpy.argsort(axis=-1).reshape(-1)[::-1][0]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = f"page_{int(max_score_page_index) + 1}"

# 将 questions 对象，保存至 文件 中
with open("./data/01_tfidf_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# 9.再获取 top10 页码
for index, question_normalize in enumerate(questions_normalize):
    # 计算点积
    score = question_normalize @ pages_content_normalize.T

    # 转换为 numpy.ndarray
    score_numpy = score.toarray()

    # argsort() 排序，获取top10
    max_score_pages_index = score_numpy.argsort(axis=-1).reshape(-1)[::-1][:10]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = [f"page_{page_index + 1}" for page_index in max_score_pages_index]

# 将 questions 对象，保存至 文件 中
with open("./data/01_tfidf_question_top10.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

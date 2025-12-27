import pdfplumber
import json
from sentence_transformers import SentenceTransformer

# 使用 SBERT 模型，进行向量化检索

# 1.读取 question 问题数据
with open("./data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 2.读取 pdf 每页文本内容
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

# 3.获取 question文本内容 和 pages_content文本内容
# SBERT 模型，句子编码，无需分词处理（模型中自带分词器）
questions_text = [question["question"] for question in questions]
pages_content_text = [page_content["content"] for page_content in pages_content]

# 4.加载 SBERT 模型，对句子进行编码
model = SentenceTransformer("../models/BAAI/bge-small-zh-v1.5")

# normalize_embeddings：对编码后的句子向量 进行 L2 归一化操作
questions_encode = model.encode(questions_text, normalize_embeddings=True)
pages_content_encode = model.encode(pages_content_text, normalize_embeddings=True)

# 5.计算每个问题对应的向量，与所有pages_content_encode之间的相似性得分，获取top1
for index, question_encode in enumerate(questions_encode):
    # 计算点积（因为进行了归一化操作，因此点积结果就是 余弦相似性）
    score = question_encode @ pages_content_encode.T

    # score 直接就是numpy.ndarray类型，无需再进行toarray()转换
    # 获取 top1
    max_score_page_index = score.argsort(axis=-1).reshape(-1)[::-1][0]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = f"page_{max_score_page_index + 1}"

# 将 questions 对象，保存至 文件 中
with open("./data/03_bge_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# 6.计算每个问题对应的向量，与所有pages_content_encode之间的相似性得分，获取top10
for index, question_encode in enumerate(questions_encode):
    # 计算点积（因为进行了归一化操作，因此点积结果就是 余弦相似性）
    score = question_encode @ pages_content_encode.T

    # score 直接就是numpy.ndarray类型，无需再进行toarray()转换
    # 获取 top10
    max_score_page_index = score.argsort(axis=-1).reshape(-1)[::-1][:10]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = [f"page_{page_index + 1}" for page_index in max_score_page_index]

# 将 questions 对象，保存至 文件 中
with open("./data/03_bge_question_top10.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# # 7.计算每个问题对应的向量，与所有pages_content_encode之间的相似性得分，获取top1（使用 bge模型 计算）
# for index, question_encode in enumerate(questions_encode):
#     # 通过 bge模型对象，计算相似性得分
#     score = model.similarity(question_encode, pages_content_encode)
#
#     # score为 torch.Tensor 类型
#     # 获取 top1
#     max_score_page_index = score.argmax(dim=-1)
#
#     # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
#     questions[index]["reference"] = f"page_{max_score_page_index.item() + 1}"
#
# # 将 questions 对象，保存至 文件 中
# with open("./data/03_bge_question_top111.json", "w", encoding="utf-8") as f:
#     json.dump(questions, f, indent=4, ensure_ascii=False)

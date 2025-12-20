import pdfplumber
import json
from sentence_transformers import SentenceTransformer

# 与上一个BERT模型向量检索一致，不同点在于对 pdf每页内容 进行chunk（分段保存）

# 1.读取 question.json 信息
with open("./data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 2.读取 pdf 文件信息
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
pages = pdf.pages

pages_content = []
# 循环读取 每一页文本信息，并进行chunk
for page_i, page in enumerate(pages):
    text = page.extract_text()

    # 对 text 进行chunk，size为40
    text_chunk = [text[i: i + 40] for i in range(0, len(text), 40)]

    # 对 chunk 后的文本列表，进行遍历
    for chunk_i, chunk in enumerate(text_chunk):
        # 将每一块 存放到pages_content中
        pages_content.append({
            "page": page_i + 1,  # 具体在pdf哪一页
            "chunk": chunk_i + 1,  # 具体在某一页的哪一chunk
            "content": chunk  # chunk文本内容
        })

# 3.获取 questions 和 pages_content 文本内容列表（用于BERT模型编码）
questions_text = [question["question"] for question in questions]
pages_content_text = [page_text["content"] for page_text in pages_content]

# 4.加载BERT模型（SBERT -- bge）
sbert_model = SentenceTransformer("../models/BAAI/bge-small-zh-v1.5")

# 对文本进行编码
# normalize_embeddings = True：对编码后的向量，进行归一化操作（矩阵@运算，即点积运算，就是余弦相似度）
# show_progress_bar = True：显示编码进度条
questions_encode = sbert_model.encode(questions_text, normalize_embeddings=True, show_progress_bar=True)
pages_content_encode = sbert_model.encode(pages_content_text, normalize_embeddings=True, show_progress_bar=True)

# 5.对编码后的结果 计算相似度，获取top1（可以通过点积，也可以通过sbert_model模型的similarity()方法）
# 方法一：点积（模型自动计算，放在最后）
for index, question_encode in enumerate(questions_encode):
    # 点积 运算
    score = question_encode @ pages_content_encode.T

    # score -> numpy.ndarray，可以直接使用 agrmax() 获取最大分数所在索引位置
    # 获取 top1
    max_score_index = score.argmax(axis=-1).item()

    # 因为 pages_content 经过了chunk，因此max_score_page_index不一定是pdf对应的页码
    max_score_page_index = pages_content[max_score_index]["page"]
    max_score_chunk_index = pages_content[max_score_index]["chunk"]
    answer = pages_content[max_score_index]["content"]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = f"page_{max_score_page_index}, chunk_{max_score_chunk_index}"
    questions[index]["answer"] = answer

# 将 questions 对象，保存至 文件 中
with open("./data/04_bge_chunk_question_top1.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# 6.获取top10
for index, question_encode in enumerate(questions_encode):
    # 点积 运算
    score = question_encode @ pages_content_encode.T

    # 获取 top10
    max_score_index = score.argsort(axis=-1).reshape(-1)[-1:-11:-1]

    # 因为 pages_content 经过了chunk，因此max_score_page_index不一定是pdf对应的页码
    max_score_page_index = [pages_content[i]["page"] for i in max_score_index]
    max_score_chunk_index = [pages_content[i]["chunk"] for i in max_score_index]
    answer = [pages_content[i]["content"] for i in max_score_index]

    # 将检索到的页码 保存至questions的reference属性，便于后续存储到文件中
    questions[index]["reference"] = [f"page_{max_score_page_index[i]},chunk_{max_score_chunk_index[i]}" for i in
                                     range(len(max_score_index))]
    questions[index]["answer"] = answer

# 将 questions 对象，保存至 文件 中
with open("./data/04_bge_chunk_question_top10.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

# # 7.方式二：sbert_model.similarity()
# for index, question_encode in enumerate(questions_encode):
#     # similarity() 方法计算相似度
#     score = sbert_model.similarity(question_encode, pages_content_encode)
#
#     # 返回的 score 是tensor
#     max_score_index_tensor = score.argsort(dim=-1, descending=True).view(-1)[:10]
#
#     # 因为 pages_content 经过了chunk，因此max_score_page_index不一定是pdf对应的页码
#     max_score_page_index = [pages_content[i.item()]["page"] for i in max_score_index_tensor]
#     max_score_chunk_index = [pages_content[i.item()]["chunk"] for i in max_score_index_tensor]
#     answer = [pages_content[i.item()]["content"] for i in max_score_index_tensor]
#
# # 将 questions 对象，保存至 文件 中
# with open("./data/04_bge_chunk_question_auto_top10.json", "w", encoding="utf-8") as f:
#     json.dump(questions, f, indent=4, ensure_ascii=False)

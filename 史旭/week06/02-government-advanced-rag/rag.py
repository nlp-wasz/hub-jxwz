# rag 流程实现类（提取文档内容，SBERT模型编码，存储到ES向量数据库，问题检索（全文+语义），重排序，调用llm）

import pdfplumber
import torch
from openai import OpenAI

from es_api import es
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#  加载 sbert 模型
bge_model = SentenceTransformer("../../models/BAAI/bge-small-zh-v1.5")
print("***  bge_model 加载完成  ***")

#  加载 rerank 模型
tokenizer = AutoTokenizer.from_pretrained("../../models/BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("../../models/BAAI/bge-reranker-base")
rerank_model.eval()
print("***  rerank 加载完成  ***")


class Rag():
    def __init__(self):
        self.document_info_index = "document_info"
        self.document_chunk_info_index = "document_chunk_info"
        self.chunk_size = 256  # chunk 大小
        self.chunk_overlap = 20  # chunk 之间关联大小
        self.search_candidate = 50  # 候选数量
        self.search_top = 10  # 检索top10

        self.embedding_model = bge_model  # 加载 编码模型
        self.llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.llm_api_key = "sk-04ab3d7290e243dda1badc5a1d5ac858"
        self.llm_model = "qwen-plus"


    # 删除文档内容（es数据库中的信息）
    def document_delete(self, knowledge_document):
        res = es.delete_by_query(
            index=self.document_info_index,
            query={
                "term": {
                    "document_id": knowledge_document.document_id
                }
            }
        )
        es.indices.refresh(index=self.document_info_index)
        print(f"从 document_info 删除【{res['deleted']}】数据")
        print(res)

        res = es.delete_by_query(
            index=self.document_chunk_info_index,
            query={
                "term": {
                    "document_id": knowledge_document.document_id
                }
            }
        )
        es.indices.refresh(index=self.document_chunk_info_index)
        print(f"从 document_chunk_info 删除【{res['deleted']}】数据")

    # 更新文档时  更新es文档数据中 对应的信息
    def document_update(self, knowledge_document):
        res = es.update_by_query(
            index=self.document_info_index,
            query={
                "term": {
                    "document_id": knowledge_document.document_id
                }
            },
            script={  # 更新脚本
                "source": '''
                    ctx._source.document_title = params.document_title;
                    ctx._source.document_category = params.document_category;
                    ctx._source.file_path = params.file_path
                ''',
                "params": {
                    "document_title": knowledge_document.document_title,
                    "document_category": knowledge_document.document_category,
                    "file_path": knowledge_document.file_path
                }
            },
            refresh=True  # 使得更新立即可见
        )
        print(f"从 document_info 更新【{res['updated']}】数据")
        print(res)

    # 提取文档内容
    def extract_document(self, knowledge_document):
        # 获取 文档存储地址  文档类型
        file_path = knowledge_document.file_path
        file_type = knowledge_document.file_type

        if file_type == 'application/pdf':
            # 提取 pdf 文件内容
            with pdfplumber.open(file_path) as pdf:
                pages = pdf.pages
                # 将 pages 文档信息 添加到ES文档数据库中
                self.save_document_to_es(pages, knowledge_document)
        else:
            pass

    # 保存 pages 文档信息 到 es文档数据库
    def save_document_to_es(self, pages, knowledge_document):
        document_id = knowledge_document.document_id
        document_title = knowledge_document.document_title
        document_category = knowledge_document.document_category
        knowledge_id = knowledge_document.knowledge_id
        file_path = knowledge_document.file_path

        # 便利 pages，提取每一页文本信息
        for page_index, page in enumerate(pages):
            text_info = page.extract_text()

            # 先将本页信息 保存到 document_info 索引
            document_info = {
                "document_id": document_id,
                "document_title": document_title,
                "document_category": document_category,
                "knowledge_id": knowledge_id,
                "file_path": file_path,
                "abstract": text_info
            }

            es.index(index=self.document_info_index, document=document_info)
            es.indices.refresh(index=self.document_info_index)

            # 在 对每一页文档信息 进行chunk（将每一块info 存储到 document_chunk_info 索引）
            # 每一块 内容需要编码
            document_chunks = self.document_chunk(text_info, self.chunk_size, self.chunk_overlap)

            # 对 每一块内容 进行编码（语义检索是用到）
            document_chunks_encode = self.get_embedding(document_chunks)

            # 循环 存储到 es
            for chunk_index in range(len(document_chunks_encode)):
                # 获取
                document_chunk_info = {
                    "chunk_id": chunk_index + 1,
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "page_number": page_index + 1,
                    "chunk_content": document_chunks[chunk_index],
                    "embedding_vector": document_chunks_encode[chunk_index]
                }
                es.index(index=self.document_chunk_info_index, document=document_chunk_info)
                es.indices.refresh(index=self.document_chunk_info_index)

                print(f"第{page_index + 1}页，第{chunk_index + 1}块")

    # 划分 文档 chunk
    def document_chunk(self, text_info, chunk_size, overlap):
        # 记录 每一chunk块 的文本内容
        document_chunks = []
        start_index = 0
        end_index = chunk_size
        for i in range(0, len(text_info), chunk_size):
            document_chunks.append(text_info[start_index:end_index])

            # 更新 start_index 和 end_index
            start_index = end_index - overlap
            end_index = start_index + chunk_size

        return document_chunks

    # 对 每一块内容 进行编码（语义检索是用到）
    def get_embedding(self, embedding_content):
        # 编码
        return self.embedding_model.encode(embedding_content, normalize_embeddings=True, show_progress_bar=True)

    #  bm25 全文检索 获取top10
    def bm25_search(self, question, knowledge_id):
        es_search = {
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "chunk_content": question
                        }
                    },
                    "filter": {
                        "term": {
                            "knowledge_id": knowledge_id
                        }
                    }
                }
            },
            "size": self.search_top
        }

        bm25_top10 = es.search(
            index=self.document_chunk_info_index,
            body=es_search,
            fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
            source=False
        )

        return bm25_top10

    #  bge 语义检索 获取top10
    def bge_search(self, question, knowledge_id):
        # 获取 question 编码
        question_encode = self.get_embedding(question)

        bge_top10 = es.search(
            index=self.document_chunk_info_index,
            body={
                "knn": {  # 语义检索方式
                    "field": "embedding_vector",  # 检索字段（向量）
                    "query_vector": question_encode,  # 检索的向量
                    "k": self.search_top,  # top10
                    "num_candidates": self.search_candidate,  # 候选50个，从50个中选择top10
                    "filter": {
                        "term": {
                            "knowledge_id": knowledge_id
                        }
                    }
                }
            },
            fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
            source=False
        )

        return bge_top10

    # RAG 问答 流程
    def rag_chat(self, question, knowledge_id):
        # 先通过 bm25 全文检索 获取top10
        bm25_top10 = self.bm25_search(question, knowledge_id)
        print(f"bm25检索top_10：{bm25_top10}")

        # 再通过 语义检索 top10
        bge_top10 = self.bge_search(question, knowledge_id)
        print(f"bge检索top_10：{bge_top10}")

        # 多路召回 融合排序
        page_chunk_content = {}
        fusion_result = {}
        k = 60
        for bm25_index, bm25_chunk in enumerate(bm25_top10["hits"]["hits"]):
            # 获取 chunk 对应 page_number 和 chunk_id
            chunk_key = f"{bm25_chunk["fields"]["page_number"]}_{bm25_chunk["fields"]["chunk_id"]}"

            if chunk_key not in fusion_result:
                fusion_result[chunk_key] = 1 / (k + bm25_index)
            else:
                fusion_result[chunk_key] += 1 / (k + bm25_index)

            # 保存 chunk_key 对应的文本内容
            if chunk_key not in page_chunk_content:
                page_chunk_content[chunk_key] = str(bm25_chunk["fields"]["chunk_content"])

        for bge_index, bge_chunk in enumerate(bge_top10["hits"]["hits"]):
            # 获取 chunk 对应 page_number 和 chunk_id
            chunk_key = f"{bge_chunk["fields"]["page_number"]}_{bge_chunk["fields"]["chunk_id"]}"

            if chunk_key not in fusion_result:
                fusion_result[chunk_key] = 1 / (k + bge_index)
            else:
                fusion_result[chunk_key] += 1 / (k + bge_index)

                # 保存 chunk_key 对应的文本内容
                if chunk_key not in page_chunk_content:
                    page_chunk_content[chunk_key] = str(bge_chunk["fields"]["chunk_content"])

        # 对 融合后的 fusion_result 进行排序
        fusion_sorted = sorted(fusion_result.items(), key=lambda item: item[1], reverse=True)

        # 取 前五个 进行重排序（构建 重排序 inputs）  (问题， chunk内容)句子对
        question_content = []

        fusion_rerank = fusion_sorted[:5]
        for tuple_page_chunk in fusion_rerank:
            # chunk_key：page_chunk，即116_1
            chunk_key = tuple_page_chunk[0]

            # 构建 句子对（从 page_chunk_content 中获取 文本信息）
            question_content.append([question, page_chunk_content[chunk_key]])

        # 使用 重排序分词器 处理句子对
        fusion_tokenize = tokenizer(question_content, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt")

        # fusion_tokenize 包含三部分：input_ids  attention_mask  token_type_ins
        # 转换为 字典（输入参数 inputs）
        fusion_inputs = {key: value_tensor for key, value_tensor in fusion_tokenize.items()}

        # 使用 rerank_model 重新打分
        with torch.no_grad():
            rerank_result = rerank_model(**fusion_inputs).logits

            # 获取 分数最大值 对应的索引
            rerank_result = rerank_result.view(-1).float()
            max_source_index = rerank_result.argmax(-1).item()

        # 获取 对应索引位置的 chunk_key（page_chunk）
        page_chunk = fusion_rerank[max_source_index][0]

        # 构建 prompt 提示词
        prompt = '''你是一个专业的汽车领域专家，请根据下面提供的资料，回答用户提出的问题。
            回答内容必须是提供资料中的内容，如果用户提出的问题与汽车无关，请返回“无法回答”，
            如果用户提出的问题不在资料范围内，请返回“无法回答”，如果提供的资料可以回答用户的问题，请根据资料回答用户问题。
            
            资料：{0}
            
            问题：{1}
        '''.format(page_chunk_content[page_chunk], question)

        # print(prompt)

        # 调用 大模型
        llm_response = self.llm_qwen_api_openai(prompt)

        return llm_response, page_chunk_content[page_chunk]

    # 调用 llm 接口
    def llm_qwen_api_openai(self, prompt):
        openai = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key
        )

        res = openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return res.choices[0].message.content

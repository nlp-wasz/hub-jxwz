from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)
from sentence_transformers import SentenceTransformer
import torch
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


client=OpenAI(
    api_key="***",
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_markdown_langchain(directory):
    """使用LangChain加载markdown"""
    loader = DirectoryLoader(
        directory,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    #最终返回包含所有文档内容的列表
    return loader.load()

result=load_markdown_langchain('./')
print(result)


def split_markdown_documents(docs, chunk_size=500, chunk_overlap=100):
        """分割文档为小块"""
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return markdown_splitter.split_documents(docs)


def qa(query,top_k):
    template = '''
       你是一个简洁的中文助理，基于给定参考段落回答问题。"
        "请用3-5句话概括答案，引用最相关的内容。\n\n"
        f"问题：{query}\n\n参考段落：\n{content}"

       '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "content"])

    model_name='./user_data/model/Qwen3-Embedding-0.6B'
    embedder=SentenceTransformer(model_name,trust_remote_code=True)

    markdown_documents=result=load_markdown_langchain('./')
    chunks = split_markdown_documents(markdown_documents)
    chunks_text = [chunk.page_content for chunk in chunks]
    chunks_embeddings=embedder.encode_document(chunks_text)

    query_embedding = embedder.encode_query(query, convert_to_tensor=True)

    similarity_scores = embedder.similarity(query_embedding, chunks_embeddings)[0]

    # 保留最相关的知识点
    scores, indices = torch.topk(similarity_scores, k=top_k)
    info = ""
    for score, idx in zip(scores, indices):
        # print(f"(Score: {score:.4f})", corpus[idx])
        info += chunks_text[idx] + "\n"

    message = prompt.format_prompt(query=query, info=info)

    result = client.chat.completions.create(
        model="",
        messages=message
    )
    return result.choices[0].message.content



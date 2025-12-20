import streamlit as st
import hashlib
import os
# 设置环境变量以解决 Pydantic v2 兼容性问题
os.environ["LANGCHAIN_ALLOW_DUPLICATE_VALIDATORS"] = "true"
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 从 pymilvus 导入 MilvusClient
from pymilvus import MilvusClient
# 使用 SentenceTransformer 作为本地嵌入模型
from sentence_transformers import SentenceTransformer
import tempfile
import re
import numpy as np

# 加载环境变量
load_dotenv()


# 初始化模型和嵌入
@st.cache_resource
def init_resources():
    # 使用本地 BGE 模型作为嵌入模型，避免 API 费用
    embeddings = SentenceTransformer(r'D:\learning\八斗\models\bge-small-zh-v1.5')
    llm = ChatOpenAI(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return embeddings, llm


# 初始化Milvus向量数据库连接（参考Week11项目的方式）
@st.cache_resource
def init_milvus():
    # 使用与Week11项目相同的连接配置
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )
    
    # 确保集合存在
    collections = client.list_collections()
    if "w15" not in collections:
        # 定义集合模式
        schema = {
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": 512},  # BGE 模型的维度
                {"name": "text", "type": "VARCHAR", "max_length": 65535},
                {"name": "source", "type": "VARCHAR", "max_length": 256},
                {"name": "seq_num", "type": "INT64"}
            ]
        }
        # 创建集合
        client.create_collection(
            collection_name="w15",
            schema=schema
        )
        
        # 创建向量索引
        index_params = {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE"
        }
        client.create_index(
            collection_name="w15",
            index_params=index_params
        )
        
        # 加载集合
        client.load_collection(collection_name="w15")
    
    return client


# 处理上传的Markdown文件
def process_markdown_file(file_content, filename):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(file_content)

    # 为每个分割添加文件名元数据
    docs = []
    for i, split in enumerate(md_header_splits):
        doc = Document(
            page_content=split.page_content,
            metadata={
                "source": filename,
                "seq_num": i,
                **split.metadata
            }
        )
        docs.append(doc)

    return docs


# 生成文档唯一标识符
def generate_doc_id(content):
    return hashlib.md5(content.encode()).hexdigest()


# 将文档添加到Milvus数据库
def add_documents_to_milvus(client, embeddings, docs):
    # 为文档生成嵌入向量
    texts = [doc.page_content for doc in docs]
    embeddings_list = embeddings.encode(texts)
    
    # 准备插入到Milvus的数据
    data = []
    for i, (doc, embedding) in enumerate(zip(docs, embeddings_list)):
        data.append({
            "vector": embedding.tolist(),  # 转换为列表
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "seq_num": doc.metadata.get("seq_num", 0)
        })
    
    # 插入数据到Milvus
    result = client.insert(
        collection_name="w15",
        data=data
    )
    
    # 刷新集合以确保数据可见
    client.flush(collection_name="w15")
    
    return result


# 从Milvus检索相关文档
def search_documents(client, embeddings, query, top_k=4):
    # 生成查询向量
    query_embedding = embeddings.encode(query).tolist()
    
    # 执行搜索
    results = client.search(
        collection_name="w15",
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source", "seq_num"]
    )
    
    # 处理搜索结果
    docs = []
    for result in results[0]:
        doc = Document(
            page_content=result["entity"]["text"],
            metadata={
                "source": result["entity"]["source"],
                "seq_num": result["entity"]["seq_num"]
            }
        )
        docs.append(doc)
    
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.set_page_config(page_title="Markdown RAG问答系统", layout="wide")
    st.title("📘 Markdown文档RAG问答系统")

    # 侧边栏
    st.sidebar.header("操作面板")

    # 初始化资源
    embeddings, llm = init_resources()
    milvus_client = init_milvus()

    # 文件上传
    uploaded_files = st.sidebar.file_uploader(
        "上传Markdown文件",
        type=["md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("正在处理文件..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode('utf-8')
                docs = process_markdown_file(content, uploaded_file.name)
                all_docs.extend(docs)

                st.success(f"已处理文件: {uploaded_file.name} ({len(docs)} 个片段)")

            # 添加到向量数据库
            if all_docs:
                with st.spinner("正在存储到向量数据库..."):
                    result = add_documents_to_milvus(milvus_client, embeddings, all_docs)
                    st.success(f"已成功存储 {len(all_docs)} 个文档片段到向量数据库! 插入ID: {result['ids']}")

    # 显示集合信息
    try:
        collections = milvus_client.list_collections()
        if "w15" in collections:
            stats = milvus_client.get_collection_stats(collection_name="w15")
            st.sidebar.info(f"当前集合: w15")
            st.sidebar.info(f"集合实体数: {stats.get('row_count', 0)}")
        else:
            st.sidebar.info("集合 w15 尚未创建")
    except Exception as e:
        st.sidebar.warning(f"无法获取集合信息: {str(e)}")

    # 查询输入
    st.subheader("💬 询问关于文档的问题")
    query = st.text_input("请输入您的问题:", placeholder="例如: 这些文档主要讲了什么内容?")

    if query:
        with st.spinner("正在检索和生成答案..."):
            try:
                # 检索相关文档
                retrieved_docs = search_documents(milvus_client, embeddings, query)

                # 构建上下文
                context = format_docs(retrieved_docs)

                # 构建提示词
                template = """
                你是问答助手，基于以下上下文回答用户问题。
                如果上下文中没有相关信息，请说明无法从文档中找到答案。

                上下文:
                {context}

                问题: {question}

                回答:
                """

                prompt = ChatPromptTemplate.from_template(template)

                # 构建RAG链
                rag_chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # 生成回答
                response = rag_chain.invoke(query)

                # 显示回答
                st.markdown("### 🤖 回答")
                st.write(response)

                # 显示参考文档
                st.markdown("### 📚 参考文档")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"参考文档 #{i + 1}"):
                        st.markdown(f"**来源:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"**内容:**\n\n{doc.page_content}")

            except Exception as e:
                st.error(f"处理查询时出错: {str(e)}")

    # 显示使用说明
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 使用说明")
    st.sidebar.markdown("""
    1. 在左侧上传Markdown文件
    2. 系统会自动解析并存储到向量数据库
    3. 输入问题获取基于文档的回答
    4. 可以查看参考的文档片段
    """)


if __name__ == "__main__":
    main()
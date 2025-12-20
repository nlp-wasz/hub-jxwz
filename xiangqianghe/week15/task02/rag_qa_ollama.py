import os
import sys
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 配置
MODEL_NAME = "qwen3:0.6b"  # 用于生成的模型
EMBEDDING_MODEL_NAME = "qwen:0.5b" # 用于Embeddings的模型
PDF_NAME = "2507-PaddleOCR 3.0"
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
# Mineru 输出目录结构通常为 output_dir/pdf_name_without_ext/auto/pdf_name_without_ext.md
MARKDOWN_PATH = os.path.join(TASK_DIR, PDF_NAME, "auto", f"{PDF_NAME}.md")

def load_and_process_document():
    if not os.path.exists(MARKDOWN_PATH):
        print(f"错误：找不到Markdown文件: {MARKDOWN_PATH}")
        print("请确保先运行mineru解析PDF。")
        return None

    print(f"正在加载文档: {MARKDOWN_PATH}...")
    try:
        # 读取Markdown文件内容
        with open(MARKDOWN_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用RecursiveCharacterTextSplitter分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.create_documents([text])
        print(f"文档已分割为 {len(texts)} 个块。")
        return texts
    except Exception as e:
        print(f"加载文档时出错: {e}")
        return None

def setup_rag_system(texts):
    print(f"正在初始化Embeddings ({EMBEDDING_MODEL_NAME})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    print("正在构建向量索引 (FAISS)...")
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("向量索引构建完成。")
        return vectorstore
    except Exception as e:
        print(f"构建向量索引时出错: {e}")
        return None

def run_qa_loop(vectorstore):
    print(f"正在初始化LLM ({MODEL_NAME})...")
    llm = Ollama(model=MODEL_NAME)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    print("\n" + "="*50)
    print("RAG问答系统已准备就绪！")
    print("输入 'quit', 'exit' 或 'q' 退出。")
    print("="*50 + "\n")
    
    if len(sys.argv) > 1:
        # 非交互模式：直接回答命令行参数中的问题
        query = " ".join(sys.argv[1:])
        print(f"收到问题: {query}")
        try:
            result = qa_chain.invoke({"query": query})
            answer = result['result']
            source_docs = result['source_documents']
            
            print(f"\n回答: {answer}")
            print("\n参考片段:")
            for i, doc in enumerate(source_docs, 1):
                print(f"[{i}] {doc.page_content[:100]}...")
        except Exception as e:
            print(f"处理问题时出错: {e}")
        return

    while True:
        query = input("\n请输入您的问题: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("退出系统。")
            break
        
        if not query:
            continue
            
        print("正在思考...")
        try:
            result = qa_chain.invoke({"query": query})
            answer = result['result']
            source_docs = result['source_documents']
            
            print(f"\n回答: {answer}")
            print("\n参考片段:")
            for i, doc in enumerate(source_docs, 1):
                print(f"[{i}] {doc.page_content[:100]}...")
        except Exception as e:
            print(f"处理问题时出错: {e}")

def main():
    texts = load_and_process_document()
    if texts:
        vectorstore = setup_rag_system(texts)
        if vectorstore:
            run_qa_loop(vectorstore)

if __name__ == "__main__":
    main()

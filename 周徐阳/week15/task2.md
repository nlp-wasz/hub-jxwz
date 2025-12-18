# MinerU 本地部署与 RAG 问答完整指南

MinerU 是一个强大的开源文档解析工具，能够将 PDF/Word 等文档转换为高质量的 Markdown 格式，非常适合用于 RAG 应用。下面是完整的部署和使用教程。

---

## 一、MinerU 本地部署

### 1. 环境要求

```bash
# Python 版本要求
Python >= 3.9, < 3.13

# 推荐使用 conda 创建虚拟环境
conda create -n mineru python=3.10
conda activate mineru
```

### 2. 安装 MinerU

```bash
# 方式一：pip 安装（推荐）
pip install magic-pdf[full] -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二：从源码安装
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .[full]
```

### 3. 下载模型文件

MinerU 需要下载 OCR 和布局检测模型：

```bash
# 使用官方脚本下载模型
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/PDF-Extract-Kit-1.0', local_dir='./models')"
```

### 4. 配置文件设置

创建配置文件 `magic-pdf.json`（通常在用户目录下）：

```json
{
  "models-dir": "/path/to/your/models",
  "device-mode": "cuda",  // 或 "cpu"
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "formula-config": {
    "mfd_model": "yolo_v8_mfd",
    "mfr_model": "unimernet_small"
  },
  "table-config": {
    "model": "tablemaster",
    "enable": true
  }
}
```

---

## 二、使用 MinerU 解析文档

### 1. 命令行方式

```bash
# 解析 PDF 文件
magic-pdf -p your_document.pdf -o ./output

# 指定解析模式
# auto: 自动选择（推荐）
# ocr: 强制OCR模式
# txt: 文本模式
magic-pdf -p your_document.pdf -o ./output -m auto
```

### 2. Python API 方式

```python
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
import os
import json

def parse_pdf(pdf_path, output_dir):
    """解析PDF文档"""
    
    # 读取PDF文件
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出writer
    image_writer = FileBasedDataWriter(os.path.join(output_dir, "images"))
    md_writer = FileBasedDataWriter(output_dir)
    
    # 创建解析管道（自动模式）
    pipe = UNIPipe(pdf_bytes, [], image_writer)
    
    # 执行解析
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    
    # 获取Markdown内容
    md_content = pipe.pipe_mk_markdown(
        os.path.join(output_dir, "images"),
        drop_mode="none"
    )
    
    # 保存Markdown文件
    md_path = os.path.join(output_dir, "output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return md_content

# 使用示例
md_content = parse_pdf("./document.pdf", "./parsed_output")
print(md_content)
```

---

## 三、构建 RAG 问答系统

### 1. 整体架构

```
PDF/Word → MinerU解析 → Markdown文本 → 文本分块 → 向量化 → 向量数据库
                                                              ↓
用户问题 → 向量化 → 相似度检索 → 获取相关文档块 → LLM生成回答
```

### 2. 完整代码实现

```python
# ============ 安装依赖 ============
# pip install langchain langchain-community chromadb sentence-transformers openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# ============ Step 1: 文档解析 ============
def parse_document(file_path, output_dir="./parsed"):
    """使用MinerU解析文档"""
    import subprocess
    
    # 调用MinerU命令行
    cmd = f"magic-pdf -p {file_path} -o {output_dir} -m auto"
    subprocess.run(cmd, shell=True, check=True)
    
    # 读取生成的Markdown
    # MinerU会在output_dir下创建以文件名命名的子目录
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    md_path = os.path.join(output_dir, base_name, "auto", f"{base_name}.md")
    
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

# ============ Step 2: 文本分块 ============
def split_text(text, chunk_size=500, chunk_overlap=50):
    """将文本分割成块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# ============ Step 3: 创建向量数据库 ============
def create_vector_store(documents, persist_dir="./chroma_db"):
    """创建向量数据库"""
    # 使用本地embedding模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 创建Chroma向量数据库
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vector_store

# ============ Step 4: RAG问答 ============
class RAGQASystem:
    def __init__(self, vector_store, llm_api_key=None, llm_base_url=None):
        self.vector_store = vector_store
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        docs = self.vector_store.similarity_search(query, k=top_k)
        return docs
    
    def generate_answer(self, query, context_docs):
        """使用LLM生成回答"""
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""基于以下参考内容回答问题。如果参考内容中没有相关信息，请说明无法从文档中找到答案。

参考内容：
{context}

问题：{query}

回答："""
        
        # 调用LLM（这里使用OpenAI兼容接口）
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.llm_api_key or "your-api-key",
            base_url=self.llm_base_url or "https://api.openai.com/v1"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或其他模型
            messages=[
                {"role": "system", "content": "你是一个专业的文档问答助手，根据提供的文档内容准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def ask(self, query, top_k=3):
        """完整的RAG问答流程"""
        # 检索
        docs = self.retrieve(query, top_k)
        
        # 生成
        answer = self.generate_answer(query, docs)
        
        return {
            "question": query,
            "answer": answer,
            "sources": [doc.page_content[:200] + "..." for doc in docs]
        }

# ============ 完整使用示例 ============
def main():
    # 1. 解析文档
    print("正在解析文档...")
    md_content = parse_document("./your_document.pdf")
    
    # 2. 文本分块
    print("正在分割文本...")
    documents = split_text(md_content)
    print(f"共分割为 {len(documents)} 个文本块")
    
    # 3. 创建向量数据库
    print("正在创建向量数据库...")
    vector_store = create_vector_store(documents)
    
    # 4. 创建RAG系统并问答
    print("RAG系统初始化完成！")
    rag_system = RAGQASystem(
        vector_store=vector_store,
        llm_api_key="your-api-key",
        llm_base_url="https://api.openai.com/v1"
    )
    
    # 交互式问答
    while True:
        query = input("\n请输入问题（输入 'quit' 退出）: ")
        if query.lower() == 'quit':
            break
        
        result = rag_system.ask(query)
        print(f"\n回答：{result['answer']}")
        print(f"\n参考来源：")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source}")

if __name__ == "__main__":
    main()
```

---

## 四、进阶优化建议

| 优化方向 | 具体方法 |
|---------|---------|
| **分块策略** | 使用语义分块，保持段落完整性 |
| **检索增强** | 混合检索（关键词+向量），重排序 |
| **Embedding** | 使用更强的模型如 `bge-large-zh` |
| **LLM** | 使用本地模型如 Qwen、ChatGLM |
| **缓存** | 对常见问题进行缓存 |

---

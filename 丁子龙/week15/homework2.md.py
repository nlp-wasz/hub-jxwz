import sys
import os

# 运行mineru解析pdf为markdown

# os.system("mineru -p 2509-MinerU2.5.pdf -o ./output/")
# 结果路径 2025-12-12_15-43.jpg
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI  # 或用其他 LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#模型下载
from modelscope import snapshot_download

# 确保已运行：ollama pull qwen3:8b  # 或 qwen2.5:7b、qwen:14b 等
Settings.llm = Ollama(
    model="qwen3:8b",        # ← 与 `ollama list` 中名称严格一致
    request_timeout=300.0,   # 大模型生成慢，调高超时
    # base_url="http://localhost:11434",  # 默认，可省略
)

print("✅ Ollama LLM [qwen3:8b] 已配置")
# 1️⃣ 显式从 ModelScope 下载模型（确保来源可控）
EMBED_MODEL_ID = "iic/gte_Qwen2-1.5B-instruct"
LOCAL_MODEL_DIR = f"./models/{EMBED_MODEL_ID}"  # → ./models/iic_nan-bee-embedding

if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"📥 正在从 ModelScope 下载模型: {EMBED_MODEL_ID} → {LOCAL_MODEL_DIR}")
    snapshot_download(
        model_id=EMBED_MODEL_ID,
        cache_dir="./models",  # 下载到 ./models/
        revision="master",
        local_files_only=False,
    )
    # snapshot_download 返回的是实际路径，例如: ./models/iic/nan-bee-embedding/
    # 我们统一重命名/获取
    import glob
    actual_path = glob.glob(f"./models/{EMBED_MODEL_ID.split('/')[0]}/*")[0]
    if actual_path != LOCAL_MODEL_DIR:
        os.rename(actual_path, LOCAL_MODEL_DIR)
else:
    print(f"✅ 模型已存在本地: {LOCAL_MODEL_DIR}")

# 2️⃣ 使用本地路径初始化 embedding（100% 确定来源）
Settings.embed_model = HuggingFaceEmbedding(
    model_name=LOCAL_MODEL_DIR,  # ← 关键：用本地路径
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"🎉 Embedding 模型已加载（来自本地 {LOCAL_MODEL_DIR}）")

# 1. 加载单个 markdown 文件（SimpleDirectoryReader 也支持单文件 list）
documents = SimpleDirectoryReader(input_files=["/home/dzl/baDouNLP/week/Week15/homework/output/2509-MinerU2.5/auto/2509-MinerU2.5.md"]).load_data()

# 2. 构建向量索引（默认用 OpenAI embedding，可换为本地模型如 BAAI/bge-small）
index = VectorStoreIndex.from_documents(documents)

# 3. 查询
query_engine = index.as_query_engine()
response = query_engine.query("介绍一下mineru")
print(response)
# 结果展示 2025-12-12_17-02.jpg


































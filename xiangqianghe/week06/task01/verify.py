import torch
from sentence_transformers import SentenceTransformer
import warnings


warnings.filterwarnings("ignore", message="module 'torch.sparse' has no attribute '_spsolve'")
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

# 测试sparse模块
try:
    print("_spsolve属性:", torch.sparse._spsolve)
    print("✅ PyTorch sparse模块正常")
except AttributeError as e:
    print("❌ 问题仍然存在:", e)

# 测试sentence-transformers
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode("Hello world")
    print("✅ Sentence Transformers正常工作")
    print("嵌入向量形状:", embeddings.shape)
except Exception as e:
    print("❌ Sentence Transformers错误:", e)
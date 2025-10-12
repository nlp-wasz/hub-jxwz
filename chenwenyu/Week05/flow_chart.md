flowchart TD
    A[用户提问] --> B[查询处理]
    B --> C[查询向量化<br>使用嵌入模型]
    
    subgraph KnowledgeBase [知识库处理流程]
        D[文档加载<br>PDF/TXT/MD等] --> E[文本分割<br>chunk_size=1000]
        E --> F[向量嵌入<br>Embedding模型]
        F --> G[向量存储<br>Chroma/FAISS]
    end
    
    C --> H[相似性检索<br>Top-K相关文档]
    H --> I[上下文增强<br>拼接提示词]
    
    subgraph LLM [本地大模型处理]
        J[加载本地模型<br>Qwen2.5/Mistral等] --> K[生成回答]
    end
    
    I --> K
    K --> L[返回回答<br>附带参考来源]
    
    G --> H
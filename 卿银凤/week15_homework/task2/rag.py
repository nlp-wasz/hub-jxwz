"""
RAG问答系统 - 基于东方证券大语言模型投研投顾文档
"""

import os
import numpy as np
# 禁用tokenizers并行，避免fork警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#导入必要的库
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class SimpleRAG:
    """简单的RAG问答系统"""
    
    def __init__(self, doc_path: str, embedding_model: str = "../BAAI/bge-small-zh-v1.5"):
        """
        初始化RAG系统
        
        Args:
            doc_path: 文档路径
            embedding_model: 向量模型名称
        """
        self.doc_path = doc_path
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
        
        # 加载向量模型
        print("正在加载向量模型...")
        self.encoder = SentenceTransformer(embedding_model)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "sk-be3c9c14e12046f59f6e0c5f9bad8fbf"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        
        # 加载并处理文档
        self._load_and_process_document()
    
    def _load_and_process_document(self):
        """加载并处理文档"""
        print(f"正在加载文档: {self.doc_path}")
        
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 文档分块
        self.chunks = self._split_document(content)
        print(f"文档已分割为 {len(self.chunks)} 个片段")
        
        # 生成向量
        print("正在生成向量...")
        self.embeddings = self.encoder.encode(self.chunks, normalize_embeddings=True)
        print("向量生成完成")
    
    def _split_document(self, content: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        将文档分割成块
        
        Args:
            content: 文档内容
            chunk_size: 每块大小
            overlap: 重叠大小
        
        Returns:
            分块列表
        """
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 跳过图片引用
            if para.startswith('![') and para.endswith(')'):
                continue
            
            # 如果当前块加上新段落不超过限制，则合并
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        检索相关文档片段
        
        Args:
            query: 查询问题
            top_k: 返回top k个结果
        
        Returns:
            (片段, 相似度) 列表
        """
        # 计算查询向量
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # 获取top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        基于上下文生成回答
        
        Args:
            query: 用户问题
            context: 检索到的上下文
        
        Returns:
            生成的回答
        """
        prompt = f"""你是一个专业的金融投研助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实说明。

参考资料：
{context}

用户问题：{query}

请给出准确、专业的回答："""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "qwen-max"),
                messages=[
                    {"role": "system", "content": "你是一个专业的金融投研助手，擅长分析投研投顾相关内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    def ask(self, query: str, top_k: int = 3) -> dict:
        """
        RAG问答
        
        Args:
            query: 用户问题
            top_k: 检索片段数量
        
        Returns:
            包含答案和检索结果的字典
        """
        # 检索相关片段
        retrieved = self._retrieve(query, top_k)
        
        # 构建上下文
        context = "\n\n---\n\n".join([chunk for chunk, _ in retrieved])
        
        # 生成回答
        answer = self._generate_answer(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved
        }


def main():
    # 文档路径
    doc_path = os.path.join(
        os.path.dirname(__file__),
        "full.md"
    )
    
    # 初始化RAG系统
    rag = SimpleRAG(doc_path)
    
    # 示例问题
    test_questions = [
        "什么是智能投顾？",
        "投资建议？",
        "茅台在南京地区的销售情况?",
        "投研投顾应用案例?",
        "投研工作主要包括哪些内容？"
    ]
    
    print("\n" + "=" * 60)
    print("RAG问答系统测试")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        
        result = rag.ask(question)
        print(f"回答: {result['answer']}")
        
        print("\n检索到的相关片段:")
        for i, (chunk, score) in enumerate(result['retrieved_chunks'], 1):
            print(f"  [{i}] 相似度: {score:.4f}")
            print(f"      内容: {chunk[:100]}...")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

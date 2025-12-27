import os
from typing import List, Optional


class SimpleRAGSystem:
    def __init__(self):
        self.agent_dir = "./agent/auto/"
        self.agent_file = os.path.join(self.agent_dir, "agent.md")

        # 确保目录存在
        os.makedirs(self.agent_dir, exist_ok=True)

    def load_document_content(self) -> Optional[str]:
        """
        加载本地agent.md文件内容
        """
        try:
            with open(self.agent_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Agent file not found at {self.agent_file}")
            print("Please make sure the file exists in the specified location.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def simple_rag_search(self, question: str, content: str, top_k: int = 3) -> List[str]:
        """
        简单的RAG搜索实现
        """
        # 将内容按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        # 简单的关键词匹配（实际应用中可以使用更复杂的相似度算法）
        relevant_paragraphs = []
        question_words = set(question.lower().split())

        for para in paragraphs:
            para_words = set(para.lower().split())
            # 计算简单的重叠度
            overlap = len(question_words.intersection(para_words))
            if overlap > 0:
                relevant_paragraphs.append((overlap, para))

        # 按相关性排序并返回top_k
        relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)
        return [para for _, para in relevant_paragraphs[:top_k]]

    def answer_question(self, question: str) -> str:
        """
        基于本地文档回答问题
        """
        content = self.load_document_content()
        if not content:
            return "Error: No document content available. Please check if agent.md exists."

        relevant_info = self.simple_rag_search(question, content)

        if not relevant_info:
            return "Sorry, I couldn't find relevant information in the document to answer your question."

        # 构建回答
        answer = f"Based on the document, here's what I found:\n\n"
        for i, info in enumerate(relevant_info, 1):
            answer += f"{i}. {info}\n\n"

        return answer


# 使用示例
def main():
    # 创建RAG系统实例
    rag_system = SimpleRAGSystem()

    # 检查文档是否存在
    if not os.path.exists(rag_system.agent_file):
        print(f"Warning: Agent file not found at {rag_system.agent_file}")
        print("Please ensure the file exists before asking questions.")

    # 示例问答
    questions = [
        "What is the main topic of this document?",
        "What methods were used in the research?",
        "What are the key findings?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag_system.answer_question(question)
        print(f"Answer: {answer}")
        print("-" * 50)

    # 交互式问答
    print("\nEntering interactive mode. Type 'quit' to exit.")
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break

        answer = rag_system.answer_question(user_question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
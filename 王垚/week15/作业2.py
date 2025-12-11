import os
import re
import json
import time
import numpy as np
import pandas as pd
import warnings
import faiss
import markdown
import faiss
# 导入库
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class RAGSystem:
    """RAG问答系统主类"""

    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        self.client = None

    def setup_environment(self):

        # 初始化模型
        print("📥 正在加载BGE embedding模型...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        print(f"✅ 模型加载完成！向量维度: {self.embedding_model.get_sentence_embedding_dimension()}")

        # 初始化API客户端
        if 'OPENAI_API_KEY' not in os.environ:
            api_key = input("请输入DeepSeek API密钥: ")
            os.environ['OPENAI_API_KEY'] = api_key
            os.environ['OPENAI_BASE_URL'] = 'https://api.deepseek.com'

        self.client = OpenAI()
        print("✅ DeepSeek API客户端初始化完成！")

        return True

    def load_document(self, file_path):
        """加载文档"""
        print(f"📄 正在加载文档: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 清理文本
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = '\n'.join(line.strip() for line in content.split('\n'))
            content = content.strip()

            print(f"✅ 文档加载成功！长度: {len(content)} 字符")
            return content

        except Exception as e:
            print(f"❌ 文档加载失败: {e}")
            return None

    def chunk_text(self, text, chunk_size=512, chunk_overlap=50):
        """文本分块"""
        print("🔪 正在进行智能文本分块...")

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 检查添加段落后是否会超过块大小
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(test_chunk) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'metadata': {
                            'length': len(current_chunk),
                            'paragraph_count': len(current_chunk.split('\n\n'))
                        }
                    })

                # 开始新块
                current_chunk = paragraph

        # 处理最后一个块
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'metadata': {
                    'length': len(current_chunk),
                    'paragraph_count': len(current_chunk.split('\n\n'))
                }
            })

        print(f"✅ 分块完成！共 {len(chunks)} 个文档块")

        # 显示统计信息
        chunk_lengths = [chunk['metadata']['length'] for chunk in chunks]
        print(f"📊 分块统计:")
        print(f"   平均长度: {np.mean(chunk_lengths):.1f} 字符")
        print(f"   最大长度: {max(chunk_lengths)} 字符")
        print(f"   最小长度: {min(chunk_lengths)} 字符")

        self.chunks = chunks
        return chunks

    def build_vector_index(self):
        """构建向量索引"""
        print("🔍 正在构建向量索引...")

        if not self.chunks:
            print("❌ 请先进行文本分块！")
            return False

        # 提取文本
        texts = [chunk['text'] for chunk in self.chunks]

        # 生成向量
        start_time = time.time()
        print("📊 正在生成文本向量...")
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        end_time = time.time()

        print(f"✅ 向量生成完成！耗时: {end_time - start_time:.2f}秒")
        print(f"   向量维度: {self.embeddings.shape}")

        # 创建FAISS索引

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        self.index.add(self.embeddings.astype('float32'))

        print(f"✅ FAISS索引构建完成！索引大小: {self.index.ntotal} 向量")
        return True

    def search_similar(self, query, k=5):
        """搜索相似内容"""
        if self.index is None:
            print("❌ 请先构建向量索引！")
            return []

        # 生成查询向量
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(score),
                    'index': int(idx)
                })

        return results

    def generate_answer(self, question, max_context_length=2000):
        """生成答案"""
        print(f"🤔 正在回答问题: {question}")

        # 检索相关内容
        search_results = self.search_similar(question, k=5)

        if not search_results:
            return {
                'question': question,
                'answer': '抱歉，在文档中没有找到与您的问题相关的信息。',
                'sources': [],
                'response_time': 0
            }

        # 构建上下文
        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            chunk_text = result['chunk']['text']
            formatted_chunk = f"[来源{i+1}]\n{chunk_text}\n"

            if current_length + len(formatted_chunk) <= max_context_length:
                context_parts.append(formatted_chunk)
                current_length += len(formatted_chunk)
            else:
                break

        context = "\n".join(context_parts)

        # 创建提示词
        prompt = f"""你是一个专业的AI技术文档助手，请基于以下提供的文档内容回答用户问题。

文档内容:
{context}

用户问题: {question}

请根据文档内容回答问题，要求:
1. 答案必须基于提供的文档内容
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 适当引用文档中的具体数据和技术细节
5. 使用中文回答

答案:"""

        # 调用API生成答案
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的AI技术文档助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"

        end_time = time.time()

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'score': result['score'],
                    'snippet': result['chunk']['text'][:200] + "..."
                }
                for result in search_results[:3]
            ],
            'response_time': end_time - start_time
        }

    def run_test(self):
        """运行测试"""
        print("\n" + "="*60)
        print("🧪 开始运行测试用例")
        print("="*60)

        test_questions = [
            "DeepSeek-V3的总参数量是多少？",
            "DeepSeek-V3使用了哪些优化技术？",
            "训练成本如何？",
            "模型的性能表现如何？"
        ]

        results = []

        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试 {i}/{len(test_questions)}: {question}")
            print("-" * 50)

            result = self.generate_answer(question)
            results.append(result)

            print(f"💬 答案: {result['answer']}")
            print(f"⏱️  响应时间: {result['response_time']:.2f}秒")

            if result['sources']:
                print(f"📚 相关来源:")
                for j, source in enumerate(result['sources']):
                    print(f"   {j+1}. 相似度: {source['score']:.4f}")

        print(f"\n{'='*60}")
        print("✅ 测试完成！")
        print(f"📊 总体统计:")
        print(f"   平均响应时间: {np.mean([r['response_time'] for r in results]):.2f}秒")
        print(f"   最快响应时间: {min([r['response_time'] for r in results]):.2f}秒")
        print(f"   最慢响应时间: {max([r['response_time'] for r in results]):.2f}秒")
        print("="*60)

        return results

    def interactive_mode(self):
        """交互模式"""
        print("\n🚀 DeepSeek-V3技术报告问答系统")
        print("="*50)
        print("输入您的问题，输入 'quit' 退出")
        print("="*50)

        while True:
            try:
                query = input("\n请输入您的问题: ").strip()

                if query.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用！")
                    break

                elif not query:
                    print("请输入有效的问题！")
                    continue

                result = self.generate_answer(query)

                print(f"\n💬 答案: {result['answer']}")
                print(f"⏱️  响应时间: {result['response_time']:.2f}秒")

            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    print("🎯 RAG问答系统 - DeepSeek-V3技术报告")
    print("="*60)

    # 初始化系统
    rag_system = RAGSystem()

    # 环境配置
    if not rag_system.setup_environment():
        print("❌ 环境配置失败，请检查依赖包安装")
        return

    # 加载文档
    doc_path = "2412-DeepSeek-V3.md"
    if not os.path.exists(doc_path):
        print(f"❌ 文档文件不存在: {doc_path}")
        print("请确保DeepSeek-V3技术报告文档在当前目录下")
        return

    content = rag_system.load_document(doc_path)
    if not content:
        return

    # 文本分块
    rag_system.chunk_text(content)

    # 构建向量索引
    if not rag_system.build_vector_index():
        return

    # 运行测试
    test_results = rag_system.run_test()

    # 询问是否进入交互模式
    user_input = input("\n是否进入交互问答模式？(y/n): ").strip().lower()
    if user_input in ['y', 'yes', '是']:
        rag_system.interactive_mode()

    print("\n🎉 RAG问答系统运行完成！")


if __name__ == "__main__":
    main()
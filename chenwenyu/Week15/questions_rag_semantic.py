import numpy as np
import json
import re
from typing import List, Dict, Tuple
import hashlib
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer

class SemanticRAGSystem:
    def __init__(self, model_name='/root/autodl-tmp/models/google-bert/bert-base-chinese', 
                 cache_dir='./rag_cache'):
        """
        基于语义的RAG系统
        
        参数:
            model_name: 使用的句子转换模型
            cache_dir: 缓存目录
        """        
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化模型
        print(f"加载模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # 数据存储
        self.questions = []  # 原始问题列表
        self.question_ids = []  # 问题ID列表
        self.question_texts = []  # 问题文本列表
        self.embeddings = None  # 问题嵌入向量
        self.index = None  # FAISS索引
        
        # 缓存文件路径
        self.cache_hash = None
        self.embeddings_cache_path = os.path.join(cache_dir, 'embeddings.npy')
        self.metadata_cache_path = os.path.join(cache_dir, 'metadata.pkl')
        
    def parse_markdown(self, md_content: str) -> List[Tuple[int, str]]:
        """
        解析Markdown格式的问题
        """
        # 匹配数字开头的问题
        pattern = r'^(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)'
        matches = re.findall(pattern, md_content, re.MULTILINE | re.DOTALL)
        
        questions = []
        for match in matches:
            q_id = int(match[0])
            q_text = match[1].strip()
            # 清理文本，移除换行符和多余空格
            q_text = re.sub(r'\s+', ' ', q_text)
            questions.append((q_id, q_text))
        
        print(f"解析到 {len(questions)} 个问题")
        return questions
    
    def generate_content_hash(self, questions: List[Tuple[int, str]]) -> str:
        """
        生成内容哈希，用于缓存管理
        """
        content_str = ''.join([f"{q_id}{text}" for q_id, text in questions])
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def load_from_markdown(self, md_content: str = None, md_file_path: str = None):
        """
        从Markdown加载问题
        """
        if md_file_path:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        
        # 解析问题
        questions = self.parse_markdown(md_content)
        self.questions = questions
        self.question_ids = [q[0] for q in questions]
        self.question_texts = [q[1] for q in questions]
        
        # 生成内容哈希
        self.cache_hash = self.generate_content_hash(questions)
        
        # 检查缓存
        if self._check_cache():
            print("使用缓存数据...")
            self._load_from_cache()
        else:
            print("生成新的嵌入向量...")
            self._generate_embeddings()
            self._build_faiss_index()
            self._save_to_cache()
    
    def _check_cache(self) -> bool:
        """
        检查缓存是否有效
        """
        if not os.path.exists(self.metadata_cache_path):
            return False
        
        try:
            with open(self.metadata_cache_path, 'rb') as f:
                metadata = pickle.load(f)
                if metadata.get('hash') != self.cache_hash:
                    return False
                if metadata.get('model') != self.model_name:
                    return False
            return os.path.exists(self.embeddings_cache_path)
        except:
            return False
    
    def _load_from_cache(self):
        """从缓存加载"""
        # 加载嵌入向量
        self.embeddings = np.load(self.embeddings_cache_path)
        
        # 加载元数据
        with open(self.metadata_cache_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 构建FAISS索引
        self._build_faiss_index()
        
        print(f"从缓存加载完成，维度: {self.embeddings.shape}")
    
    def _save_to_cache(self):
        """保存到缓存"""
        # 保存嵌入向量
        np.save(self.embeddings_cache_path, self.embeddings)
        
        # 保存元数据
        metadata = {
            'hash': self.cache_hash,
            'model': self.model_name,
            'count': len(self.questions),
            'dimension': self.embeddings.shape[1],
            'question_ids': self.question_ids
        }
        
        with open(self.metadata_cache_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"缓存已保存: {self.cache_hash}")
    
    def _generate_embeddings(self):
        """生成问题嵌入向量"""
        print("生成嵌入向量...")
        
        # 分批处理，避免内存问题
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(self.question_texts), batch_size):
            batch_texts = self.question_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化以便余弦相似度计算
            )
            embeddings_list.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"已处理: {min(i+batch_size, len(self.question_texts))}/{len(self.question_texts)}")
        
        self.embeddings = np.vstack(embeddings_list)
        print(f"嵌入向量生成完成，形状: {self.embeddings.shape}")
    
    def _build_faiss_index(self):
        """构建FAISS索引用于快速搜索"""
        print("构建FAISS索引...")
        
        dimension = self.embeddings.shape[1]
        
        # 使用内积索引（因为向量已经归一化，内积=余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)
        
        # 添加到索引
        self.index.add(self.embeddings)
        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
    
    def semantic_search(self, query: str, top_k: int = 10, 
                        threshold: float = 0.3) -> List[Dict]:
        """
        语义搜索相关问题
        
        参数:
            query: 查询文本
            top_k: 返回最相关的k个问题
            threshold: 相似度阈值
        
        返回:
            相关问题的列表，按相似度排序
        """
        if self.index is None or len(self.questions) == 0:
            raise ValueError("请先加载问题数据")
        
        # 生成查询嵌入向量
        query_embedding = self.model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 搜索最相似的top_k个向量
        similarities, indices = self.index.search(query_embedding, min(top_k * 2, len(self.questions)))
        
        # 筛选和格式化结果
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity < threshold or idx >= len(self.questions):
                continue
            
            q_id = self.question_ids[idx]
            q_text = self.question_texts[idx]
            
            results.append({
                'rank': len(results) + 1,
                'original_id': q_id,
                'question': q_text,
                'similarity': float(similarity),
                'index': int(idx)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     semantic_weight: float = 0.7) -> List[Dict]:
        """
        混合搜索：结合语义和关键词
        
        参数:
            semantic_weight: 语义搜索权重 (0-1)
        """
        # 语义搜索结果
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # 关键词搜索结果（简单实现）
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # 合并结果
        combined_scores = {}
        
        # 处理语义结果
        for result in semantic_results:
            q_id = result['original_id']
            combined_scores[q_id] = {
                'score': result['similarity'] * semantic_weight,
                'semantic_score': result['similarity'],
                'keyword_score': 0,
                'question': result['question'],
                'original_id': q_id
            }
        
        # 处理关键词结果
        for result in keyword_results:
            q_id = result['original_id']
            keyword_score = result.get('score', 0) / 100  # 归一化
            
            if q_id in combined_scores:
                combined_scores[q_id]['score'] += keyword_score * (1 - semantic_weight)
                combined_scores[q_id]['keyword_score'] = keyword_score
            else:
                combined_scores[q_id] = {
                    'score': keyword_score * (1 - semantic_weight),
                    'semantic_score': 0,
                    'keyword_score': keyword_score,
                    'question': result['question'],
                    'original_id': q_id
                }
        
        # 按综合分数排序
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # 添加排名
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1
        
        return sorted_results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        基于关键词的搜索（作为语义搜索的补充）
        """
        # 简单的TF-IDF风格关键词匹配
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored_results = []
        for idx, (q_id, text) in enumerate(self.questions):
            text_terms = set(re.findall(r'\w+', text.lower()))
            
            # 计算Jaccard相似度
            intersection = query_terms.intersection(text_terms)
            union = query_terms.union(text_terms)
            
            if union:
                jaccard_sim = len(intersection) / len(union)
            else:
                jaccard_sim = 0
            
            # 计算词频分数
            term_freq_score = 0
            for term in query_terms:
                if term in text.lower():
                    term_freq_score += 1
            
            # 综合分数
            score = jaccard_sim * 0.7 + (term_freq_score / len(query_terms)) * 0.3 if query_terms else 0
            
            if score > 0:
                scored_results.append({
                    'original_id': q_id,
                    'question': text,
                    'score': score * 100,
                    'index': idx
                })
        
        # 排序并返回top_k
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]
    
    def find_similar_questions(self, question_id: int, top_k: int = 5) -> List[Dict]:
        """
        查找与指定问题相似的问题
        """
        if question_id not in self.question_ids:
            raise ValueError(f"问题ID {question_id} 不存在")
        
        idx = self.question_ids.index(question_id)
        question_embedding = self.embeddings[idx:idx+1]
        
        # 搜索相似问题
        similarities, indices = self.index.search(question_embedding, top_k + 1)
        
        results = []
        for i, (similarity, neighbor_idx) in enumerate(zip(similarities[0], indices[0])):
            # 跳过自身
            if neighbor_idx == idx:
                continue
            
            if neighbor_idx < len(self.questions):
                q_id = self.question_ids[neighbor_idx]
                q_text = self.question_texts[neighbor_idx]
                
                results.append({
                    'rank': len(results) + 1,
                    'original_id': q_id,
                    'question': q_text,
                    'similarity': float(similarity),
                    'compared_to': question_id
                })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def cluster_questions(self, n_clusters: int = 10):
        """
        对问题进行聚类
        """
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # 组织聚类结果
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            q_id = self.question_ids[idx]
            q_text = self.question_texts[idx]
            
            if label not in clusters:
                clusters[label] = {
                    'size': 0,
                    'questions': [],
                    'centroid': kmeans.cluster_centers_[label]
                }
            
            clusters[label]['questions'].append({
                'id': q_id,
                'text': q_text,
                'distance_to_centroid': float(np.linalg.norm(
                    self.embeddings[idx] - kmeans.cluster_centers_[label]
                ))
            })
            clusters[label]['size'] += 1
        
        # 为每个聚类提取关键词
        for label, cluster_data in clusters.items():
            # 这里可以添加关键词提取逻辑
            cluster_data['representative_questions'] = sorted(
                cluster_data['questions'],
                key=lambda x: x['distance_to_centroid']
            )[:3]
        
        return clusters
    
    def export_results(self, results: List[Dict], output_format: str = 'markdown') -> str:
        """
        导出搜索结果
        
        参数:
            output_format: 'markdown', 'json', 'text'
        """
        if output_format == 'markdown':
            output = f"## 搜索结果 (共 {len(results)} 个)\n\n"
            for result in results:
                output += f"**{result['rank']}. [原题{result['original_id']}]** {result['question']}\n"
                if 'similarity' in result:
                    output += f"   相似度: {result['similarity']:.3f}\n"
                if 'score' in result:
                    output += f"   综合分数: {result['score']:.3f}\n"
                output += "\n"
            return output
        
        elif output_format == 'json':
            return json.dumps(results, ensure_ascii=False, indent=2)
        
        else:  # text
            output = f"搜索结果 (共 {len(results)} 个):\n\n"
            for result in results:
                output += f"{result['rank']}. [原题{result['original_id']}] {result['question']}\n"
            return output
    
    def interactive_search(self):
        """
        交互式搜索界面
        """
        print("=" * 60)
        print("语义RAG系统 - 面试题搜索")
        print("=" * 60)
        print("命令:")
        print("  /s <关键词>    - 语义搜索")
        print("  /h <关键词>    - 混合搜索")
        print("  /k <关键词>    - 关键词搜索")
        print("  /sim <问题ID>  - 查找相似问题")
        print("  /cluster       - 聚类分析")
        print("  /quit          - 退出")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n请输入命令: ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['/quit', '/exit', '/q']:
                    print("再见！祝面试成功！")
                    break
                
                elif command.startswith('/s '):
                    query = command[3:].strip()
                    if query:
                        print(f"\n语义搜索: '{query}'")
                        results = self.semantic_search(query, top_k=10)
                        print(self.export_results(results, 'text'))
                
                elif command.startswith('/h '):
                    query = command[3:].strip()
                    if query:
                        print(f"\n混合搜索: '{query}'")
                        results = self.hybrid_search(query, top_k=10)
                        print(self.export_results(results, 'text'))
                
                elif command.startswith('/k '):
                    query = command[3:].strip()
                    if query:
                        print(f"\n关键词搜索: '{query}'")
                        results = self.keyword_search(query, top_k=10)
                        print(self.export_results(results, 'text'))
                
                elif command.startswith('/sim '):
                    try:
                        q_id = int(command[5:].strip())
                        print(f"\n查找与问题 {q_id} 相似的问题:")
                        results = self.find_similar_questions(q_id, top_k=5)
                        print(self.export_results(results, 'text'))
                    except ValueError:
                        print("请输入有效的问题ID")
                
                elif command == '/cluster':
                    print("\n正在进行问题聚类...")
                    clusters = self.cluster_questions(n_clusters=10)
                    
                    print(f"聚类完成，共 {len(clusters)} 个类别:")
                    for label, data in clusters.items():
                        print(f"\n类别 {label} (包含 {data['size']} 个问题):")
                        for q in data['representative_questions'][:2]:
                            print(f"  - [问题{q['id']}] {q['text'][:80]}...")
                
                else:
                    print("未知命令，请输入 /s, /h, /k, /sim, /cluster 或 /quit")
            
            except KeyboardInterrupt:
                print("\n再见！祝面试成功！")
                break
            except Exception as e:
                print(f"错误: {e}")


# 使用示例
def main():
    # 示例Markdown内容
    sample_md = """
1. 请解释什么是词向量，以及它在自然语言处理中的主要作用。
2. 简述“稀疏词向量”（如one-hot编码）的原理，并分析其优缺点。
3. Word2Vec有哪两种模型？请分别解释它们的原理。
4. 解释GloVe词向量的原理，与Word2Vec相比有何优缺点？
5. 什么是BERT？它在自然语言处理中有哪些应用？
6. 解释Transformer模型中的自注意力机制。
7. 什么是迁移学习？在NLP中如何应用？
8. 请解释LSTM和GRU的区别。
9. 什么是梯度消失和梯度爆炸？如何解决？
10. 解释Dropout在神经网络中的作用。
11. 什么是过拟合？如何防止过拟合？
12. 解释正则化在机器学习中的作用。
13. 什么是交叉验证？为什么使用交叉验证？
14. 请解释准确率、精确率、召回率和F1分数的区别。
15. 什么是ROC曲线和AUC值？
16. 解释PCA主成分分析的原理。
17. 什么是支持向量机（SVM）？
18. 解释K-means聚类算法。
19. 什么是决策树？如何构建决策树？
20. 解释随机森林的原理。
    """
    
    # 初始化语义RAG系统
    print("初始化语义RAG系统...")
    rag = SemanticRAGSystem(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',  # 支持中文的模型
        cache_dir='./interview_rag_cache'
    )
    
    # 加载数据
    rag.load_from_markdown(md_content=sample_md)
    
    # 测试搜索
    test_queries = [
        "词向量和嵌入",
        "神经网络优化",
        "机器学习评估指标",
        "文本分类模型"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: '{query}'")
        print(f"{'='*60}")
        
        # 语义搜索
        results = rag.semantic_search(query, top_k=3)
        print("语义搜索结果:")
        for result in results:
            print(f"  [{result['rank']}] (相似度: {result['similarity']:.3f}) "
                  f"[原题{result['original_id']}] {result['question'][:60]}...")
    
    # 启动交互式界面
    # rag.interactive_search()


# 针对265个问题的优化版本
class OptimizedSemanticRAG(SemanticRAGSystem):
    def __init__(self, **kwargs):
        # 使用更强大的模型
        kwargs['model_name'] = kwargs.get('model_name', 
                                         '/root/autodl-tmp/models/google-bert/bert-base-chinese')
        super().__init__(**kwargs)
        
        # 增强的停用词列表
        self.stop_words = set([
            '请', '解释', '什么', '是', '以及', '在', '中', '的', '主要', '作用',
            '简述', '和', '其', '分析', '优缺点', '如', '并', '它', '如何',
            '为什么', '区别', '原理', '有哪些'
        ])
    
    def enhanced_semantic_search(self, query: str, top_k: int = 10,
                                 use_query_expansion: bool = True) -> List[Dict]:
        """
        增强的语义搜索，支持查询扩展
        """
        # 查询扩展：添加相关术语
        expanded_query = query
        if use_query_expansion:
            expanded_query = self.expand_query(query)
            print(f"扩展查询: {query} -> {expanded_query}")
        
        # 执行语义搜索
        results = self.semantic_search(expanded_query, top_k=top_k)
        
        # 重排序：考虑查询和问题的长度比例
        for result in results:
            query_len = len(query)
            question_len = len(result['question'])
            length_ratio = min(query_len, question_len) / max(query_len, question_len)
            
            # 调整相似度分数
            result['enhanced_similarity'] = result['similarity'] * 0.8 + length_ratio * 0.2
        
        # 按增强后的相似度重新排序
        results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        
        # 更新排名
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def expand_query(self, query: str) -> str:
        """
        查询扩展：添加相关术语
        """
        # 简单的中文同义词扩展
        synonym_dict = {
            '词向量': ['词嵌入', 'word embedding', '向量表示'],
            '神经网络': ['深度学习', '神经网络模型', '神经网络算法'],
            'bert': ['bert模型', '预训练模型', 'transformer'],
            'transformer': ['自注意力', 'attention机制'],
            'lstm': ['长短期记忆网络', '循环神经网络'],
            'svm': ['支持向量机', '支持向量机模型'],
            'pca': ['主成分分析', '降维'],
            '过拟合': ['过拟合问题', '模型过拟合'],
            '大模型':['LLM','大型语言模型','大语言模型'],
        }
        
        expanded_terms = [query]
        for term, synonyms in synonym_dict.items():
            if term in query.lower():
                expanded_terms.extend(synonyms[:2])  # 添加最多2个同义词
        
        return ' '.join(expanded_terms)
    
    def save_search_history(self, query: str, results: List[Dict], 
                           filepath: str = 'search_history.json'):
        """
        保存搜索历史
        """
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_count': len(results),
            'top_results': results[:3]
        }
        
        # 加载现有历史
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # 添加新记录
        history.append(history_entry)
        
        # 保存（只保留最近100条）
        if len(history) > 100:
            history = history[-100:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def batch_search(self, queries: List[str], output_file: str = 'batch_results.md'):
        """
        批量搜索
        """
        all_results = []
        
        for query in queries:
            print(f"处理查询: {query}")
            results = self.enhanced_semantic_search(query, top_k=5)
            
            query_results = {
                'query': query,
                'results': results
            }
            all_results.append(query_results)
            
            # 保存中间结果
            self.save_search_history(query, results)
        
        # 导出所有结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 批量搜索结果\n\n")
            for qr in all_results:
                f.write(f"## 查询: {qr['query']}\n\n")
                for result in qr['results']:
                    f.write(f"{result['rank']}. [原题{result['original_id']}] "
                           f"{result['question']} (相似度: {result['enhanced_similarity']:.3f})\n")
                f.write("\n")
        
        print(f"批量搜索结果已保存到: {output_file}")
        return all_results


# 主程序
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='语义RAG系统')
    parser.add_argument('--md_file', type=str, help='Markdown文件路径')
    parser.add_argument('--query', type=str, help='查询语句')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    # 使用优化版RAG系统
    rag = OptimizedSemanticRAG(
        model_name='/root/autodl-tmp/models/google-bert/bert-base-chinese',
        cache_dir='./interview_semantic_cache'
    )
    
    if args.md_file:
        print(f"从文件加载: {args.md_file}")
        rag.load_from_markdown(md_file_path=args.md_file)
    else:
        print("使用示例数据...")
        main()
        exit()

    if args.interactive:
        # 交互模式
        rag.interactive_search()
    elif args.query:
        # 执行搜索
        results = rag.enhanced_semantic_search(args.query, top_k=10)
        print(rag.export_results(results, 'markdown'))
    else:
        # 示例查询
        print("\n示例搜索:")
        queries = ["词向量模型", "深度学习优化", "机器学习评估"]
        for query in queries:
            print(f"\n查询: '{query}'")
            results = rag.enhanced_semantic_search(query, top_k=3)
            for result in results:
                print(f"  {result['rank']}. [原题{result['original_id']}] "
                      f"{result['question'][:60]}... "
                      f"(相似度: {result['enhanced_similarity']:.3f})")
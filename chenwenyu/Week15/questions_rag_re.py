import re
import json

class RegexRAGSystem:
    def __init__(self, md_content=None, md_file_path=None):
        """
        初始化RAG系统
        
        参数:
            md_content: Markdown格式的文本内容
            md_file_path: 或提供Markdown文件路径
        """
        self.questions_dict = {}
        self.question_pattern = r'(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)'
        
        if md_file_path:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        
        if md_content:
            self.parse_markdown(md_content)
    
    def parse_markdown(self, md_content):
        """解析Markdown文件，提取问题"""
        # 使用正则表达式匹配所有问题
        matches = re.findall(self.question_pattern, md_content, re.DOTALL)
        
        for match in matches:
            q_id = int(match[0])
            q_content = match[1].strip()
            # 存储问题和ID
            self.questions_dict[q_id] = q_content
    
    def extract_keywords(self, text):
        """提取关键词（简单版本）"""
        # 移除标点符号，转换为小写
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        # 分割单词
        words = text_clean.split()
        # 过滤停用词（简单版本）
        stop_words = {'请', '解释', '什么', '是', '以及', '在', '中', '的', '主要', '作用', 
                     '简述', '和', '其', '分析', '优缺点', '如', '并', '它'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return set(keywords)
    
    def calculate_relevance(self, question, keywords):
        """计算问题和关键词的相关性"""
        score = 0
        question_lower = question.lower()
        
        for keyword in keywords:
            # 使用正则表达式确保匹配完整单词
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, question_lower)
            score += len(matches) * 2  # 完整单词匹配得2分
            
            # 部分匹配（包含关系）
            if keyword in question_lower:
                score += 1
        
        return score
    
    def search_questions(self, query, top_k=5, threshold=1):
        """
        搜索相关问题
        
        参数:
            query: 查询关键词
            top_k: 返回最相关的k个问题
            threshold: 相关性阈值
        """
        # 提取查询关键词
        query_keywords = self.extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # 计算每个问题的相关性分数
        scored_questions = []
        
        for q_id, question in self.questions_dict.items():
            relevance_score = self.calculate_relevance(question, query_keywords)
            
            if relevance_score >= threshold:
                scored_questions.append({
                    'id': q_id,
                    'question': question,
                    'score': relevance_score,
                    'matched_keywords': [kw for kw in query_keywords if kw in question.lower()]
                })
        
        # 按分数降序排序
        scored_questions.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_k个结果
        return scored_questions[:top_k]
    
    def format_output(self, results, query):
        """格式化输出结果"""
        if not results:
            return f"没有找到与 '{query}' 相关的问题。"
        
        output = f"与 '{query}' 相关的面试题（共找到 {len(results)} 个）:\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['question']}\n"
            output += f"   原编号: {result['id']}, "
            output += f"相关性分数: {result['score']}, "
            output += f"匹配关键词: {', '.join(result['matched_keywords'][:3])}\n\n"
        
        return output
    
    def advanced_search(self, query, use_regex=False, category=None):
        """
        高级搜索功能
        
        参数:
            query: 查询词
            use_regex: 是否使用正则表达式查询
            category: 问题类别（可根据问题内容自动分类）
        """
        results = []
        
        if use_regex:
            # 使用正则表达式直接搜索
            try:
                pattern = re.compile(query, re.IGNORECASE)
                for q_id, question in self.questions_dict.items():
                    if pattern.search(question):
                        results.append({
                            'id': q_id,
                            'question': question,
                            'score': 10,  # 正则匹配给高分
                            'matched_keywords': ['正则匹配']
                        })
            except re.error:
                # 如果正则表达式无效，回退到普通搜索
                results = self.search_questions(query)
        else:
            results = self.search_questions(query)
        
        return results
    
    def save_index(self, filepath):
        """保存索引到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.questions_dict, f, ensure_ascii=False, indent=2)
    
    def load_index(self, filepath):
        """从JSON文件加载索引"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.questions_dict = json.load(f)

# 使用示例
def main():
    # 示例Markdown内容（实际使用时应读取您的265个问题的文件）
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
    """
    
    # 初始化RAG系统
    rag = RegexRAGSystem(md_content=sample_md)
    
    # 示例搜索
    queries = [
        "词向量",
        "神经网络",
        "BERT模型",
        "注意力机制"
    ]
    
    for query in queries:
        print("=" * 60)
        results = rag.search_questions(query, top_k=3)
        output = rag.format_output(results, query)
        print(output)
    
    # 使用正则表达式高级搜索
    print("=" * 60)
    print("高级搜索（使用正则表达式）:")
    print("=" * 60)
    
    # 搜索包含"向量"或"编码"的问题
    regex_query = r"(向量|编码)"
    results = rag.advanced_search(regex_query, use_regex=True)
    output = rag.format_output(results, regex_query)
    print(output)

# 针对265个问题文件的完整解决方案
def build_full_system(md_file_path):
    """
    构建完整的RAG系统
    
    参数:
        md_file_path: 包含265个面试题的Markdown文件路径
    """
    
    # 1. 初始化系统
    rag = RegexRAGSystem(md_file_path=md_file_path)
    
    # 2. 创建索引文件（可选）
    rag.save_index('interview_questions_index.json')
    
    # 3. 交互式搜索界面
    def interactive_search():
        print("面试题RAG系统已启动！")
        print("输入关键词搜索相关问题，输入'quit'退出")
        print("-" * 50)
        
        while True:
            query = input("\n请输入搜索关键词: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！祝面试成功！")
                break
            
            if not query:
                continue
            
            # 搜索相关问题
            results = rag.search_questions(query, top_k=10)
            
            if results:
                print(f"\n找到 {len(results)} 个相关问题:")
                print("-" * 50)
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. [原题{result['id']}] {result['question'][:100]}...")
                    print(f"   匹配度: {result['score']}, 关键词: {', '.join(result['matched_keywords'][:3])}")
                    print()
            else:
                print("未找到相关问题，请尝试其他关键词。")
    
    return rag, interactive_search

# 增强版RAG系统（支持更多功能）
class EnhancedRegexRAG(RegexRAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_category_index()
    
    def build_category_index(self):
        """根据问题内容自动分类"""
        self.categories = {
            '词向量': ['词向量', 'word2vec', 'glove', 'embedding'],
            '深度学习': ['神经网络', 'cnn', 'rnn', 'lstm', 'gru', 'transformer'],
            'bert': ['bert', 'transformer', '预训练'],
            '基础概念': ['什么', '解释', '简述', '原理', '优缺点'],
            '编程': ['代码', '实现', 'python', '编程'],
        }
        
        self.question_categories = {}
        for q_id, question in self.questions_dict.items():
            question_lower = question.lower()
            categories = []
            for category, keywords in self.categories.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        categories.append(category)
                        break
            self.question_categories[q_id] = list(set(categories))
    
    def search_by_category(self, category):
        """按类别搜索问题"""
        results = []
        for q_id, categories in self.question_categories.items():
            if category in categories:
                results.append({
                    'id': q_id,
                    'question': self.questions_dict[q_id],
                    'score': 5,  # 类别匹配基础分
                    'category': category
                })
        return results

if __name__ == "__main__":
    # 运行示例
    #main()
    
    # 实际使用时：
    # 1. 准备您的265个问题的MD文件
    # 2. 运行以下代码：
    rag_system, search_func = build_full_system("./NLP_面试题_fixed.md")
    search_func()  # 启动交互式搜索
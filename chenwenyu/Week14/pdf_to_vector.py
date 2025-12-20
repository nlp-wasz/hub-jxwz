import os
import re
import json
import numpy as np
import pandas as pd
import pdfplumber
from typing import Dict, List, Tuple, Optional
import hashlib
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class FormulaInfo:
    """存储公式信息的类"""
    pdf_name: str
    background_text: str
    background_vector: np.ndarray
    formula_latex: str
    formula_description: str
    parameters: Dict[str, str]
    file_hash: str

class PDFFormulaExtractor:
    """PDF公式提取器 - 使用pdfplumber"""
    
    def __init__(self, model_name: str = '../../../models/google-bert/bert-base-chinese/'):
        """
        初始化提取器
        
        Args:
            model_name: 文本嵌入模型名称
        """
        self.USE_SBERT = True
        self.pdfplumber = pdfplumber
        self.formula_database = []
        self.background_vectors = []
        self.pdf_names = []

        # 初始化文本嵌入模型
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ SentenceTransformer模型加载成功: {model_name}")
        except ImportError:
            print("警告: 未安装sentence-transformers, 将使用TF-IDF作为备选")
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=512)
                self.embedding_dim = 512
            except ImportError:
                print("警告: 也未安装scikit-learn, 将使用简单词频统计")
                self.embedding_dim = 100
        
        print(f"向量维度: {self.embedding_dim}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF中提取文本 - 使用pdfplumber
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本
        """
        
        text = ""
        try:
            # 使用pdfplumber打开PDF
            with self.pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 提取页面文本
                    page_text = page.extract_text()
                    if page_text:
                        # 添加页面分隔标记
                        #text += f"\n=== 第{page_num+1}页 ===\n{page_text}"
                        text += f"\n{page_text}"

            if not text.strip():
                print(f"⚠️  警告: {pdf_path} 可能没有可提取的文本内容")
                
        except Exception as e:
            print(f"读取PDF文件 {pdf_path} 失败: {e}")
            raise
        
        return text
    
    def extract_formula_and_background(self, text: str) -> Tuple[str, str, Dict[str, str]]:
        """
        从文本中提取建模背景和公式
        
        Args:
            text: PDF提取的文本
            
        Returns:
            (background_text, formula_latex, parameters)
        """
        # 分割文本为建模背景和建模公式两部分
        parts = re.split(r'建模公式|Modeling Formula', text)
        
        if len(parts) < 2:
            # 如果没有明确的标题分割，尝试其他分割方式
            parts = re.split(r'\n\s*\n', text)
            background_text = parts[0] if len(parts) > 0 else ""
            formula_section = parts[1] if len(parts) > 1 else text
        else:
            background_text = parts[0].replace('建模背景', '').strip()
            formula_section = parts[1]
        
        # 提取LaTeX公式（假设公式在$$或$之间）
        latex_patterns = [
            r'\$\$(.*?)\$\$',  # $$公式$$
            r'\\\[(.*?)\\\]',  # \[公式\]
            r'\\begin\{equation\}(.*?)\\end\{equation\}',  # \begin{equation}公式\end{equation}
            r'\$(.*?)\$'  # $公式$
        ]
        
        formula_latex = ""
        for pattern in latex_patterns:
            matches = re.findall(pattern, formula_section, re.DOTALL)
            if matches:
                formula_latex = matches[0].strip()
                break
        
        # 如果没有找到LaTeX公式，尝试提取公式描述
        if not formula_latex:
            # 查找包含DO(t), sin, cos, exp等数学表达式的行
            math_pattern = r'[𝐷𝐷OOD][OoОО]\([tT]\)\s*=\s*[^。；;]+?[。；;]'
            matches = re.findall(math_pattern, formula_section)
            if matches:
                formula_desc = matches[0]
                # 尝试转换为LaTeX
                formula_latex = self._convert_to_latex(formula_desc)
            else:
                formula_latex = "未找到明确公式"
        
        # 提取参数描述
        parameters = self._extract_parameters(formula_section)
        
        # 清理背景文本
        background_text = self._clean_text(background_text)
        
        return background_text, formula_latex, parameters
    
    def _convert_to_latex(self, formula_desc: str) -> str:
        """将公式描述转换为LaTeX格式"""
        # 常见的替换规则
        replacements = {
            '𝐷𝑂': 'DO',
            'DO': 'DO',
            'sin': '\\sin',
            'cos': '\\cos',
            'exp': '\\exp',
            'e^': 'e^{',
            '·': '\\cdot',
            '×': '\\times',
            '÷': '\\div',
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'θ': '\\theta',
            'π': '\\pi',
            'λ': '\\lambda',
            '∑': '\\sum',
            '∫': '\\int',
            '√': '\\sqrt',
            '∞': '\\infty'
        }
        
        latex_formula = formula_desc
        for ch, repl in replacements.items():
            latex_formula = latex_formula.replace(ch, repl)
        
        return latex_formula
    
    def _extract_parameters(self, formula_section: str) -> Dict[str, str]:
        """从公式部分提取参数描述"""
        parameters = {}
        
        # 查找参数描述模式（如：• $ a $ 表示...）
        param_patterns = [
            r'[•·*]\s*\$?\s*([a-zA-Zα-ω])\s*\$?\s*[:：]?\s*([^。；\n]+)[。；\n]',
            r'([a-zA-Zα-ω])\s*表示\s*([^。；\n]+)[。；\n]',
            r'([a-zA-Zα-ω])\s*为\s*([^。；\n]+)[。；\n]'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, formula_section)
            for match in matches:
                param, desc = match
                parameters[param.strip()] = desc.strip()
        
        return parameters
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除多余空格和换行"""
        if not text:
            return ""
        # 合并多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空格
        text = text.strip()
        return text
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        if hasattr(self, 'USE_SBERT') and self.USE_SBERT and hasattr(self, 'embedding_model'):
            # 使用SentenceTransformer
            vector = self.embedding_model.encode(text, convert_to_numpy=True)
        elif hasattr(self, 'tfidf_vectorizer'):
            # 使用TF-IDF（需要先拟合）
            if not hasattr(self, 'tfidf_fitted'):
                # 临时处理：使用简单的词频统计
                words = text.lower().split()
                vocab = list(set(words))
                vector = np.zeros(self.embedding_dim)
                for i, word in enumerate(words[:self.embedding_dim]):
                    vector[i] = hash(word) % 100 / 100.0
                return vector
            else:
                vector = self.tfidf_vectorizer.transform([text]).toarray()[0]
        else:
            # 简单词频统计（备选方案）
            words = re.findall(r'\w+', text.lower())
            vector = np.zeros(self.embedding_dim)
            
            for i, word in enumerate(words[:self.embedding_dim]):
                vector[i] = hash(word) % 100 / 100.0
        
        return vector
    
    def process_pdf(self, pdf_path: str) -> Optional[FormulaInfo]:
        """处理单个PDF文件"""
        try:
            print(f"处理文件: {pdf_path}")
            
            # 计算文件哈希值（用于唯一标识）
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            # 提取文本
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                print(f"  ⚠️  文件内容为空，跳过")
                return None
            
            # 提取背景和公式
            background_text, formula_latex, parameters = self.extract_formula_and_background(text)
            
            # 生成文本向量
            background_vector = self.text_to_vector(background_text)
            
            # 创建FormulaInfo对象
            formula_info = FormulaInfo(
                pdf_name=os.path.basename(pdf_path),
                background_text=background_text[:500] + "..." if len(background_text) > 500 else background_text,
                background_vector=background_vector,
                formula_latex=formula_latex,
                formula_description=self._generate_formula_description(formula_latex, parameters),
                parameters=parameters,
                file_hash=file_hash
            )
            
            print(f"  ✓ 提取成功: {formula_info.pdf_name}")
            print(f"     背景长度: {len(background_text)} 字符")
            print(f"     公式: {formula_latex[:50]}...")
            print(f"     参数: {list(parameters.keys())}")
            
            return formula_info
            
        except Exception as e:
            print(f"处理文件 {pdf_path} 时出错: {e}")
            return None
    
    def _generate_formula_description(self, formula_latex: str, parameters: Dict[str, str]) -> str:
        """生成公式的描述文本"""
        if not parameters:
            return f"公式: {formula_latex}"
        
        param_desc = ", ".join([f"{k}: {v}" for k, v in parameters.items()])
        return f"公式: {formula_latex}\n参数含义: {param_desc}"
    
    def process_directory(self, directory_path: str, pattern: str = "*.pdf") -> None:
        """处理目录下的所有PDF文件"""
        
        pdf_files = list(Path(directory_path).glob(pattern))
        pdf_files.sort()  # 按文件名排序
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        successful = 0
        for pdf_file in pdf_files:
            formula_info = self.process_pdf(str(pdf_file))
            if formula_info:
                self.formula_database.append(formula_info)
                self.background_vectors.append(formula_info.background_vector)
                self.pdf_names.append(formula_info.pdf_name)
                successful += 1
        
        # 转换为numpy数组以便检索
        if self.background_vectors:
            self.background_vectors = np.array(self.background_vectors)
        
        print(f"\n处理完成! 成功提取 {successful}/{len(pdf_files)} 个公式")
    
    def save_database(self, output_path: str = "formula_database.json") -> None:
        """保存数据库到文件"""
        data = []
        for info in self.formula_database:
            # 将numpy数组转换为列表以便JSON序列化
            data.append({
                "pdf_name": info.pdf_name,
                "background_text": info.background_text,
                "background_vector": info.background_vector.tolist() if isinstance(info.background_vector, np.ndarray) else info.background_vector,
                "formula_latex": info.formula_latex,
                "formula_description": info.formula_description,
                "parameters": info.parameters,
                "file_hash": info.file_hash
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据库已保存到: {output_path}")
    
    def load_database(self, input_path: str = "formula_database.json") -> None:
        """从文件加载数据库"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.formula_database = []
        self.background_vectors = []
        self.pdf_names = []
        
        for item in data:
            formula_info = FormulaInfo(
                pdf_name=item["pdf_name"],
                background_text=item["background_text"],
                background_vector=np.array(item["background_vector"]),
                formula_latex=item["formula_latex"],
                formula_description=item["formula_description"],
                parameters=item["parameters"],
                file_hash=item["file_hash"]
            )
            self.formula_database.append(formula_info)
            self.background_vectors.append(formula_info.background_vector)
            self.pdf_names.append(formula_info.pdf_name)
        
        if self.background_vectors:
            self.background_vectors = np.array(self.background_vectors)
        
        print(f"数据库已加载: {len(self.formula_database)} 个公式")
    
    def find_similar_formulas(self, query_text: str, top_k: int = 3) -> List[Tuple[FormulaInfo, float]]:
        """根据查询文本查找最相似的公式"""
        if not self.formula_database:
            print("数据库为空!")
            return []
        
        # 将查询文本转换为向量
        query_vector = self.text_to_vector(query_text)
        
        # 计算余弦相似度
        similarities = []
        for i, vector in enumerate(self.background_vectors):
            # 使用余弦相似度
            try:
                sim = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            except ZeroDivisionError:
                sim = 0.0
            
            similarities.append((self.formula_database[i], sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def print_formula_info(self, formula_info: FormulaInfo) -> None:
        """打印公式信息"""
        print(f"\n{'='*60}")
        print(f"PDF文件: {formula_info.pdf_name}")
        print(f"文件哈希: {formula_info.file_hash}")
        print(f"\n建模背景:")
        print(f"{formula_info.background_text[:300]}...")
        print(f"\nLaTeX公式:")
        print(f"{formula_info.formula_latex}")
        print(f"\n公式描述:")
        print(f"{formula_info.formula_description}")
        if formula_info.parameters:
            print(f"\n参数列表:")
            for param, desc in formula_info.parameters.items():
                print(f"  {param}: {desc}")
        print(f"{'='*60}")


class FormulaRetrievalSystem:
    """公式检索系统（包含向量索引）"""
    
    def __init__(self, extractor: PDFFormulaExtractor = None):
        self.extractor = extractor or PDFFormulaExtractor()
        self.index = None
        
        # 尝试使用FAISS进行高效检索
        try:
            import faiss
            self.use_faiss = True
            self.faiss = faiss
        except ImportError:
            print("未安装faiss，将使用基础检索方法")
            self.use_faiss = False
    
    def build_index(self):
        """构建向量索引"""
        if not self.extractor.background_vectors:
            print("没有可索引的向量")
            return
        
        vectors = self.extractor.background_vectors.astype('float32')
        
        if self.use_faiss and self.faiss:
            # 使用FAISS构建索引
            dimension = vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(dimension)  # 内积索引（等同于余弦相似度，因为向量已归一化）
            
            # 归一化向量（余弦相似度需要）
            self.faiss.normalize_L2(vectors)
            self.index.add(vectors)
            print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
        else:
            print("使用基础向量存储")
    
    def search(self, query_text: str, top_k: int = 3) -> List[Tuple[FormulaInfo, float]]:
        """搜索相似的公式"""
        if not self.extractor.formula_database:
            return []
        
        query_vector = self.extractor.text_to_vector(query_text)
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        if self.use_faiss and self.index and self.faiss:
            # 使用FAISS搜索
            self.faiss.normalize_L2(query_vector)
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.extractor.formula_database):
                    similarity = 1.0 - distances[0][i] / 2.0  # 转换为相似度分数
                    results.append((self.extractor.formula_database[idx], similarity))
            return results
        else:
            # 使用基础搜索
            return self.extractor.find_similar_formulas(query_text, top_k)


# 使用示例
def main():
    # 1. 初始化提取器
    print("初始化PDF公式提取器...")
    extractor = PDFFormulaExtractor()
    
    # 2. 处理PDF文件
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    pdf_directory = "./docs"  # PDF文件目录
    docs_dir = Path(pdf_directory)
    
    if not docs_dir.exists():
        print(f"目录不存在: {pdf_directory}")
        print(f"在当前目录 ({current_dir}) 中查找PDF文件...")
        pdf_directory = "."
    
    extractor.process_directory(pdf_directory, pattern="*.pdf")
    
    if not extractor.formula_database:
        print("❌ 没有提取到任何公式，程序退出")
        return
    
    # 3. 保存数据库
    extractor.save_database("formula_database.json")
    
    # 4. 初始化检索系统
    print("\n初始化检索系统...")
    retrieval_system = FormulaRetrievalSystem(extractor)
    retrieval_system.build_index()
    
    # 5. 示例查询
    test_queries = [
        "溶解氧浓度预测模型",
        "周期性变化的数学表达",
        "水产养殖环境因子",
        "非线性动力学建模"
    ]
    
    print("\n" + "="*60)
    print("示例查询测试:")
    print("="*60)
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = retrieval_system.search(query, top_k=2)
        
        if results:
            for formula_info, similarity in results:
                print(f"  相似度: {similarity:.3f} - {formula_info.pdf_name}")
                print(f"  公式: {formula_info.formula_latex[:50]}...")
        else:
            print("  未找到相关结果")
    
    # 6. 交互式查询
    print("\n" + "="*60)
    print("公式检索系统已就绪!")
    print("输入查询文本查找相关公式，输入'quit'退出")
    print("="*60)
    
    while True:
        try:
            user_query = input("\n请输入查询: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            if not user_query:
                continue
            
            results = retrieval_system.search(user_query, top_k=3)
            
            if results:
                print(f"\n找到 {len(results)} 个相关公式:")
                for i, (formula_info, similarity) in enumerate(results, 1):
                    print(f"\n{i}. [{formula_info.pdf_name}] (相似度: {similarity:.3f})")
                    print(f"   背景: {formula_info.background_text[:100]}...")
                    print(f"   公式: {formula_info.formula_latex}")
                    
                    # 显示前两个参数
                    params = list(formula_info.parameters.items())[:2]
                    if params:
                        param_str = ", ".join([f"{k}: {v[:30]}..." for k, v in params])
                        print(f"   参数: {param_str}")
            else:
                print("未找到相关公式")
                
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"查询出错: {e}")


# 生成HTML报告
def generate_html_report(extractor: PDFFormulaExtractor, output_file: str = "formula_report.html"):
    """生成HTML格式的报告"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>PDF公式提取报告</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 40px; 
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .stats {
                background: #e8f4fc;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .formula-card { 
                border: 1px solid #e0e0e0; 
                padding: 25px; 
                margin: 25px 0; 
                border-radius: 8px;
                background-color: #fff;
                transition: all 0.3s ease;
            }
            .formula-card:hover {
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            .formula-latex { 
                font-family: "Consolas", "Monaco", monospace; 
                font-size: 18px; 
                color: #c7254e;
                margin: 15px 0;
                padding: 15px;
                background-color: #f9f2f4;
                border-left: 4px solid #d63384;
                border-radius: 4px;
                overflow-x: auto;
            }
            .parameters { 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 6px;
                margin: 15px 0;
                border-left: 3px solid #6c757d;
            }
            .parameter-item {
                margin: 8px 0;
                padding: 5px 10px;
                background: white;
                border-radius: 4px;
            }
            .file-hash { 
                float: right; 
                background-color: #28a745; 
                color: white; 
                padding: 5px 15px; 
                border-radius: 20px;
                font-size: 0.9em;
            }
            .background-text {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #17a2b8;
                margin: 10px 0;
            }
            .timestamp {
                color: #6c757d;
                font-style: italic;
                margin-top: 30px;
                text-align: right;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>📄 PDF公式提取报告</h1>
            <div class="stats">
                <p><strong>提取时间:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p><strong>提取数量:</strong> """ + str(len(extractor.formula_database)) + """ 个公式</p>
            </div>
    """
    
    for i, formula_info in enumerate(extractor.formula_database, 1):
        html_content += f"""
        <div class="formula-card">
            <h3>📋 公式 {i}: {formula_info.pdf_name} <span class="file-hash">ID: {formula_info.file_hash}</span></h3>
            
            <h4>📝 建模背景:</h4>
            <div class="background-text">
                {formula_info.background_text}
            </div>
            
            <h4>🧮 LaTeX公式:</h4>
            <div class="formula-latex">
                \\[{formula_info.formula_latex}\\]
            </div>
            
            <h4>📊 参数说明:</h4>
            <div class="parameters">
        """
        
        if formula_info.parameters:
            for param, desc in formula_info.parameters.items():
                html_content += f"""
                <div class="parameter-item">
                    <strong>{param}:</strong> {desc}
                </div>
                """
        else:
            html_content += "<p>未提取到参数说明</p>"
        
        html_content += f"""
            </div>
            
            <div style="margin-top: 15px; color: #666; font-size: 0.9em;">
                <strong>公式描述:</strong> {formula_info.formula_description[:150]}...
            </div>
        </div>
        """
    
    html_content += f"""
            <div class="timestamp">
                报告生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML报告已生成: {output_file}")


# 简化版本的主程序（不使用交互式查询）
def simple_main():
    """简化版本的主程序"""
    print("=" * 70)
    print("PDF公式提取系统")
    print("=" * 70)
    
    # 1. 检查当前目录
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    
    # 2. 初始化提取器
    print("\n初始化PDF公式提取器...")
    extractor = PDFFormulaExtractor()
    
    # 3. 处理PDF文件
    print("\n查找并处理PDF文件...")
    
    # 先尝试docs目录
    docs_dir = current_dir / "docs"
    if docs_dir.exists() and docs_dir.is_dir():
        pdf_directory = str(docs_dir)
        print(f"找到docs目录: {pdf_directory}")
    else:
        pdf_directory = str(current_dir)
        print(f"使用当前目录: {pdf_directory}")
    
    # 处理PDF文件
    extractor.process_directory(pdf_directory, pattern="*.pdf")
    
    if not extractor.formula_database:
        print("❌ 没有提取到任何公式")
        return
    
    # 4. 保存数据库
    print("\n保存提取结果...")
    extractor.save_database("formula_database.json")
    
    # 5. 生成HTML报告
    #print("\n生成HTML报告...")
    #generate_html_report(extractor, "formula_report.html")
    
    print("\n" + "="*70)
    print("✅ 程序执行完成!")
    print(f"   提取公式: {len(extractor.formula_database)} 个")
    print(f"   数据库文件: formula_database.json")
    print(f"   HTML报告: formula_report.html")
    print("="*70)


if __name__ == "__main__":
    # 使用简化版本
    simple_main()
    
    # 或者使用完整交互版本
    # main()
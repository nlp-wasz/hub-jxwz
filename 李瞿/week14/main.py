"""
主程序文件，实现基于RAG筛选和MCP工具执行的智能问答系统
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re
from fastmcp import Client
from openai import OpenAI
from collections import defaultdict

# 禁用 TensorFlow 以避免依赖冲突
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# 文档中的公式描述（用于RAG检索）
FORMULA_DESCRIPTIONS = [
    {
        "id": "agricultural_yield",
        "name": "农业产量预测模型",
        "description": "基于温度、降水量、施肥量、光照时长和土壤质量指数预测作物产量。公式为: yield_prediction = base_yield × temp_factor × rainfall_factor × fertilizer_factor × sunlight_factor × soil_factor",
        "parameters": ["temp", "rainfall", "fertilizer", "sunlight", "soil_quality"],
        "tool_name": "agricultural_yield_prediction"
    },
    {
        "id": "multidimensional_evaluation",
        "name": "多维指标综合评分函数",
        "description": "用于对多维指标进行加权计算，输出统一的数值评估结果。公式为: fun(a, b, c, d, e) = 2a + 3b² - 4c + d/e + 5",
        "parameters": ["a", "b", "c", "d", "e"],
        "tool_name": "multidimensional_evaluation"
    },
    {
        "id": "sdof_system",
        "name": "单自由度系统动力响应模型",
        "description": "用于模拟结构在地震作用下的动力响应。公式为: m·ü(t) + c·u̇(t) + k·u(t) = -m·üg(t)",
        "parameters": ["m", "c", "k", "u_g"],
        "tool_name": "sdoF_system_response"
    },
    {
        "id": "chemical_reaction",
        "name": "化工反应效率预测模型",
        "description": "基于原料浓度、温度条件和压力条件预测反应效率。公式为: y = β₁x₁ + β₂x₂ + β₃x₃ + β₀",
        "parameters": ["x1", "x2", "x3"],
        "tool_name": "chemical_reaction_efficiency"
    },
    {
        "id": "fuel_consumption",
        "name": "汽车油耗预测模型",
        "description": "基于行驶速度和车辆载重预测百公里油耗。公式为: fuel_consumption = 0.05×speed + 0.002×weight + 3",
        "parameters": ["speed", "weight"],
        "tool_name": "fuel_consumption_prediction"
    },
    {
        "id": "crop_growth_rate",
        "name": "作物生长速率模型",
        "description": "基于环境因子预测作物生长速率。公式为: dW/dt = k·(1-exp(-a·temp))·(1-exp(-b·rainfall))·sunlight·(1-exp(-c·soil_nutrient))",
        "parameters": ["temp", "rainfall", "sunlight", "soil_nutrient"],
        "tool_name": "crop_growth_rate"
    },
    {
        "id": "traffic_dynamics",
        "name": "交通流速度动态模型",
        "description": "描述交通流速度随时间的变化率。公式为: dv/dt = α·(v_free - v) - β·k/q_max·v",
        "parameters": ["v_free", "k", "q_max"],
        "tool_name": "traffic_speed_dynamics"
    },
    {
        "id": "chemical_process",
        "name": "化工过程效率模型",
        "description": "基于反应温度、压力、反应物浓度、反应时间和催化剂活性预测效率。公式为: Efficiency = 1/(1+exp(-(a₁T + a₂PC + a₃ln(t+1) + a₄k²)))",
        "parameters": ["T", "P", "C", "t", "k"],
        "tool_name": "chemical_process_efficiency"
    },
    {
        "id": "cultural_influence",
        "name": "文化传播影响力增长模型",
        "description": "描述影响力随时间的动态变化。公式为: dI/dt = k·ln(t+1)·audience_growth_rate·engagement_factor",
        "parameters": ["t", "audience_growth_rate", "engagement_factor"],
        "tool_name": "cultural_influence_growth"
    },
    {
        "id": "crop_growth_dynamics",
        "name": "作物生长动态模型",
        "description": "基于土壤含水量、氮含量和温度预测作物生长速率。公式为: dG/dt = 0.05·W·(1-exp(-0.1·N))·(1-|T-25|/20)",
        "parameters": ["W", "N", "T"],
        "tool_name": "crop_growth_dynamics"
    },
    {
        "id": "integral_calculator",
        "name": "积分计算器",
        "description": "支持定积分和不定积分计算的工具。可以计算各种数学表达式的积分。",
        "parameters": ["expression", "variable", "lower_limit", "upper_limit"],
        "tool_name": "integral_calculator"
    }
]


class FormulaQAEngine:
    def __init__(self):
        # 使用 transformers 直接加载本地 BGE 模型
        model_path = r'D:\learning\八斗\models\bge-small-zh-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()  # 设置为评估模式

        # 构建公式描述的嵌入向量
        self.formula_embeddings = self._build_formula_embeddings()

    def _encode_text(self, texts):
        """使用 BGE 模型编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )

        with torch.no_grad():
            model_output = self.model(**encoded)
            # 使用 CLS token 的表示作为句向量
            embeddings = model_output.last_hidden_state[:, 0]
            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()

    def _build_formula_embeddings(self):
        """构建公式描述的嵌入向量"""
        descriptions = [formula["description"] for formula in FORMULA_DESCRIPTIONS]
        embeddings = self._encode_text(descriptions)
        return embeddings

    def retrieve_relevant_formulas(self, query, top_k=3):
        """
        使用RAG方法检索与查询最相关的公式

        Args:
            query (str): 用户查询
            top_k (int): 返回最相关的公式数量

        Returns:
            list: 相关公式列表
        """
        # 编码用户查询
        query_embedding = self._encode_text(query)

        # 计算余弦相似度 (BGE得分)
        bge_similarities = cosine_similarity(query_embedding, self.formula_embeddings)[0]

        # 计算BM25得分
        bm25_scores = self._compute_bm25_scores(query, FORMULA_DESCRIPTIONS)

        # 使用RRF融合两种得分
        fused_scores = self._rrf_fusion(bge_similarities, bm25_scores, k=60)

        # 获取最相关的公式索引
        top_indices = np.argsort(fused_scores)[-top_k:][::-1]

        # 返回最相关的公式
        relevant_formulas = []
        for idx in top_indices:
            formula = FORMULA_DESCRIPTIONS[idx].copy()
            formula["similarity"] = float(fused_scores[idx])
            relevant_formulas.append(formula)

        return relevant_formulas

    def _compute_bm25_scores(self, query, formulas):
        """
        计算BM25得分
        
        Args:
            query (str): 查询文本
            formulas (list): 公式描述列表
            
        Returns:
            list: BM25得分列表
        """
        query_terms = query.lower().split()
        scores = []

        # 统计所有文档中的词频
        doc_term_freqs = []
        doc_lengths = []
        avg_doc_length = 0

        for formula in formulas:
            desc = formula["description"].lower()
            terms = desc.split()
            term_freq = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
            doc_term_freqs.append(term_freq)
            doc_lengths.append(len(terms))

        if doc_lengths:
            avg_doc_length = sum(doc_lengths) / len(doc_lengths)

        # 计算每个文档的BM25得分
        k1 = 1.5
        b = 0.75

        for i, formula in enumerate(formulas):
            score = 0
            desc = formula["description"].lower()

            for term in query_terms:
                if term in doc_term_freqs[i]:
                    # 计算逆文档频率(IDF)
                    df = sum(1 for freqs in doc_term_freqs if term in freqs)
                    idf = np.log((len(formulas) - df + 0.5) / (df + 0.5))

                    # 计算词频组件
                    tf = doc_term_freqs[i][term]
                    doc_len = doc_lengths[i] if doc_lengths else 1
                    tf_component = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_length))

                    score += idf * tf_component

            scores.append(score)

        return scores

    def _rrf_fusion(self, bge_scores, bm25_scores, k=60):
        """
        Reciprocal Rank Fusion (RRF) 融合算法
        
        Args:
            bge_scores (list): BGE相似度得分
            bm25_scores (list): BM25得分
            k (int): RRF参数
            
        Returns:
            list: 融合后的得分
        """
        # 获取排序索引
        bge_rank_indices = np.argsort(bge_scores)[::-1]
        bm25_rank_indices = np.argsort(bm25_scores)[::-1]

        # 初始化融合得分
        fused_scores = defaultdict(float)

        # 为BGE结果打分
        for rank, idx in enumerate(bge_rank_indices):
            fused_scores[idx] += 1 / (rank + 1 + k)

        # 为BM25结果打分
        for rank, idx in enumerate(bm25_rank_indices):
            fused_scores[idx] += 1 / (rank + 1 + k)

        # 转换为列表格式
        final_scores = [fused_scores[i] for i in range(len(bge_scores))]
        return final_scores

    def _rerank_with_bge(self, query, formulas):
        """
        使用BGE模型对候选公式进行重排序
        
        Args:
            query (str): 用户查询
            formulas (list): 候选公式列表
            
        Returns:
            list: 重排序后的公式列表
        """
        # 构造重排序任务的输入
        pairs = []
        for formula in formulas:
            text = f"查询: {query}\n公式: {formula['description']}"
            pairs.append(text)

        # 编码文本对
        embeddings = self._encode_text(pairs)

        # 计算查询与各公式的相似度
        query_embedding = self._encode_text(query)
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # 根据相似度重排序
        ranked_indices = np.argsort(similarities)[::-1]
        reranked_formulas = [formulas[i] for i in ranked_indices]

        return reranked_formulas

    async def execute_formula(self, formula_info, parameters):
        """
        通过MCP客户端执行指定的公式计算
        
        Args:
            formula_info (dict): 公式信息
            parameters (dict): 参数字典
            
        Returns:
            dict: 计算结果
        """
        try:
            # 在使用时创建MCP客户端并建立连接
            async with Client("http://localhost:8888/mcp") as client:
                tool_name = formula_info["tool_name"]
                print(f"正在调用工具 {tool_name}...")
                print(f"参数: {parameters}")
                result = await client.call_tool(tool_name, parameters)

                # 对于积分计算器，直接返回结果文本
                if tool_name == "integral_calculator":
                    return {"result": result.content, "unit": ""}
                else:
                    return {"result": result.structured_content["result"], "unit": self._get_unit(formula_info["id"])}
        except Exception as e:
            return {"error": str(e)}

    def _get_unit(self, formula_id):
        """获取公式结果的单位"""
        units = {
            "agricultural_yield": "吨/公顷",
            "multidimensional_evaluation": "评分",
            "sdof_system": "位移响应",
            "chemical_reaction": "效率值",
            "fuel_consumption": "L/100km",
            "crop_growth_rate": "kg/天",
            "traffic_dynamics": "km/h/s",
            "chemical_process": "效率(0-1)",
            "cultural_influence": "影响力/天",
            "crop_growth_dynamics": "生物量/天"
        }
        return units.get(formula_id, "")

    def extract_parameters_with_qwen(self, query, formula_info):
        """
        使用阿里云Qwen大模型从用户查询中提取参数
        
        Args:
            query (str): 用户查询
            formula_info (dict): 公式信息
            
        Returns:
            dict: 提取到的参数字典
        """
        # 构造提示词
        prompt = f"""
你是一个专业的参数提取助手。请从用户的查询中提取计算所需的参数。

公式名称: {formula_info['name']}
公式描述: {formula_info['description']}
所需参数: {', '.join(formula_info['parameters'])}

用户查询: "{query}"

请按照以下JSON格式返回参数值:
{{
    "参数名1": 数值,
    "参数名2": 数值,
    ...
}}

如果某些参数无法从查询中提取，请将其值设置为null。
对于积分计算器，特别注意以下几点：
1. expression: 数学表达式，如"x**2 + 2*x + 1"，注意幂运算符是**不是^
2. variable: 积分变量，如"x"
3. lower_limit: 积分下限（可选，对于定积分需要提供）
4. upper_limit: 积分上限（可选，对于定积分需要提供）

示例：
用户说"计算x^2在0到10上的定积分"，你应该返回：
{{
    "expression": "x**2",
    "variable": "x",
    "lower_limit": 0,
    "upper_limit": 10
}}

用户说"计算sin(x)的不定积分"，你应该返回：
{{
    "expression": "sin(x)",
    "variable": "x",
    "lower_limit": null,
    "upper_limit": null
}}

只返回JSON，不要包含其他内容。
"""

        try:
            # 创建OpenAI客户端实例
            client = OpenAI(
                api_key="sk-4c44ef4112a04e65910dfdd56774f084",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 调用阿里云Qwen模型
            completion = client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            # 解析模型返回的结果
            result = completion.choices[0].message.content.strip()
            # 提取JSON部分
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result[json_start:json_end]
                try:
                    parameters = json.loads(json_str)
                    # 确保所有必需的参数都在结果中
                    for param in formula_info["parameters"]:
                        if param not in parameters:
                            parameters[param] = None
                    return parameters
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    return {param: None for param in formula_info["parameters"]}
            else:
                print("模型返回结果格式不正确")
                return {param: None for param in formula_info["parameters"]}
        except Exception as e:
            print(f"调用模型时发生错误: {e}")
            return {param: None for param in formula_info["parameters"]}

    def extract_parameters_from_query(self, query, formula_info):
        """
        从用户查询中提取参数（默认方法）
        
        Args:
            query (str): 用户查询
            formula_info (dict): 公式信息
            
        Returns:
            dict: 提取到的参数字典，缺失的参数为None
        """
        # 对于积分计算器，使用特殊处理
        if formula_info["tool_name"] == "integral_calculator":
            return self._extract_integral_params(query)

        # 将查询转换为小写便于匹配
        query_lower = query.lower()

        # 初始化参数字典
        parameters = {param: None for param in formula_info["parameters"]}

        # 定义常见的参数关键词映射
        param_keywords = {
            "temp": ["温度", "气温", "temp", "temperature"],
            "rainfall": ["降水", "降雨", "雨水", "rainfall", "rain"],
            "fertilizer": ["肥料", "施肥", "fertilizer"],
            "sunlight": ["光照", "阳光", "日照", "sunlight", "sun"],
            "soil_quality": ["土壤质量", "土质", "soil quality", "soil"],
            "a": ["a值", "参数a", "a"],
            "b": ["b值", "参数b", "b"],
            "c": ["c值", "参数c", "c"],
            "d": ["d值", "参数d", "d"],
            "e": ["e值", "参数e", "e"],
            "m": ["质量", "重量", "mass", "weight", "m"],
            "c": ["阻尼", "damping", "c"],
            "k": ["刚度", "stiffness", "k"],
            "u_g": ["地震", "acceleration", "ground motion", "u_g"],
            "x1": ["浓度", "concentration", "x1"],
            "x2": ["温度", "temp", "temperature", "x2"],
            "x3": ["压力", "pressure", "x3"],
            "speed": ["速度", "速率", "speed", "velocity"],
            "weight": ["重量", "载重", "weight"],
            "soil_nutrient": ["养分", "营养", "nutrient", "soil nutrient"],
            "v_free": ["自由流速度", "free speed", "v_free"],
            "q_max": ["最大通行能力", "capacity", "q_max"],
            "T": ["温度", "temp", "temperature", "T"],
            "P": ["压力", "压强", "pressure", "P"],
            "C": ["浓度", "concentration", "C"],
            "t": ["时间", "time", "t"],
            "W": ["含水量", "水分", "water", "moisture", "W"],
            "N": ["氮含量", "氮", "nitrogen", "N"]
        }

        # 使用正则表达式尝试提取数字
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', query)
        number_index = 0

        # 遍历每个参数
        for param in formula_info["parameters"]:
            # 查找特定参数的关键词
            found = False
            if param in param_keywords:
                for keyword in param_keywords[param]:
                    if keyword in query_lower:
                        # 在关键词后查找数字
                        pattern = rf'{keyword}.*?([-+]?\d*\.\d+|\d+)'
                        match = re.search(pattern, query_lower)
                        if match:
                            parameters[param] = float(match.group(1))
                            found = True
                            break

            # 如果没找到特定关键词，按顺序分配数字
            if not found and number_index < len(numbers):
                parameters[param] = float(numbers[number_index])
                number_index += 1

        return parameters

    def _extract_integral_params(self, query):
        """
        专门为积分计算器提取参数
        
        Args:
            query (str): 用户查询
            
        Returns:
            dict: 积分参数字典
        """
        import re

        # 初始化参数
        params = {
            "expression": None,
            "variable": None,
            "lower_limit": None,
            "upper_limit": None
        }

        # 先尝试提取积分限
        # 匹配"从0到10"、"[0,10]"、"0->10"等格式
        limit_patterns = [
            r'(?:从|在)?(\d+\.?\d*)到(\d+\.?\d*)',
            r'(?:从|在)?(\d+\.?\d*)-(\d+\.?\d*)',
            r'(?:从|在)?(\d+\.?\d*)\s*->\s*(\d+\.?\d*)',
            r'\[(\d+\.?\d*),(\d+\.?\d*)\]',
            r'\((\d+\.?\d*),(\d+\.?\d*)\)'
        ]

        lower_limit = None
        upper_limit = None

        for pattern in limit_patterns:
            match = re.search(pattern, query)
            if match:
                lower_limit = float(match.group(1))
                upper_limit = float(match.group(2))
                break

        # 如果没有找到积分限，尝试单独匹配数字（只在明确提到定积分时）
        if lower_limit is None and upper_limit is None and ('定积分' in query or '从' in query or '到' in query):
            numbers = re.findall(r'(\d+\.?\d*)', query)
            if len(numbers) >= 2:
                # 假设前两个数字是积分限
                lower_limit = float(numbers[0])
                upper_limit = float(numbers[1])

        params["lower_limit"] = lower_limit
        params["upper_limit"] = upper_limit

        # 表达式提取
        expression = None

        # 尝试各种模式提取表达式
        expr_patterns = [
            r'计算(.+?)的(定|不定)?积分',
            r'求(.+?)的(定|不定)?积分',
            r'积分(.+?)(?:从|在|上|范围|$)',
            r'integrate\s+(.+?)(?:\s+from|$)'
        ]

        for pattern in expr_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                expression = match.group(1).strip()
                break

        # 如果还是没找到，尝试更通用的匹配
        if not expression:
            # 查找最常见的数学表达式模式
            math_patterns = [
                r'([a-zA-Z][\w\+\-\*\/\^\(\)\s\d]*)',  # 以字母开头的表达式
                r'([x\w\+\-\*\/\^\(\)\s\d]+)'  # 包含x的表达式
            ]

            for pattern in math_patterns:
                match = re.search(pattern, query)
                if match and len(match.group(1).strip()) > 1:  # 至少2个字符
                    expression = match.group(1).strip()
                    break

        if expression:
            # 特殊处理：针对"计算x^2在0到10上的定积分"这类查询
            if '在' in expression and ('到' in expression or '从' in expression):
                # 尝试清理表达式中的积分限信息
                expression = re.sub(r'在\d+到\d+上?', '', expression)
                expression = re.sub(r'在\d+从\d+', '', expression)

            # 清理表达式
            expression = re.sub(r'[\s\u3000]+', '', expression)  # 去除空格
            # 将 ^ 转换为 **
            expression = expression.replace('^', '**')
            # 去除末尾可能的介词或限定词
            expression = re.sub(r'(?:从|在|上|中|范围)?(?:\d+到\d+)?$', '', expression)
            expression = expression.strip()

            # 如果表达式中仍有积分限信息，进一步清理
            if lower_limit is not None and upper_limit is not None:
                limit_str1 = str(int(lower_limit)) if lower_limit.is_integer() else str(lower_limit)
                limit_str2 = str(int(upper_limit)) if upper_limit.is_integer() else str(upper_limit)
                expression = re.sub(f'{limit_str1}到{limit_str2}', '', expression)
                expression = re.sub(f'{limit_str1}-{limit_str2}', '', expression)
                expression = re.sub(f'{limit_str1}从{limit_str2}', '', expression)

            params["expression"] = expression.strip() or "x**2"  # 默认表达式

            # 尝试提取变量（通常是最常见的字母）
            letters = re.findall(r'[a-zA-Z]', expression)
            if letters:
                # 优先选择x作为变量（最常见）
                if 'x' in letters:
                    params["variable"] = 'x'
                else:
                    # 否则选择第一个字母
                    params["variable"] = letters[0]
            else:
                # 默认变量为x
                params["variable"] = "x"
        else:
            # 默认设置
            params["expression"] = "x**2"  # 默认表达式
            params["variable"] = "x"

        # 特殊处理：如果只有下限没有上限，可能是解析错误
        if params["lower_limit"] is not None and params["upper_limit"] is None:
            params["upper_limit"] = params["lower_limit"]
            params["lower_limit"] = 0.0

        return params

    def get_missing_parameters(self, parameters):
        """
        获取缺失的参数列表
        
        Args:
            parameters (dict): 参数字典
            
        Returns:
            list: 缺失的参数名列表
        """
        return [param for param, value in parameters.items() if value is None]


async def main():
    # 初始化问答引擎
    engine = FormulaQAEngine()

    # 获取API密钥
    api_key = input("请输入阿里云DashScope API密钥（留空则使用默认的参数提取方法）: ").strip()
    if api_key:
        # 如果提供了API密钥，则使用Qwen大模型提取参数
        use_qwen = True
        print("已启用阿里云Qwen大模型参数提取")
    else:
        use_qwen = False
        print("使用默认参数提取方法")

    print("=== 文档公式智能问答系统 ===")
    print("请输入您的问题，系统将自动匹配相关公式并进行计算")
    print("输入 'quit' 退出系统")
    print("示例积分问题: 计算x^2在0到1上的定积分")

    while True:
        query = input("\n请输入您的问题: ").strip()

        if query.lower() == 'quit':
            print("感谢使用，再见！")
            break

        if not query:
            continue

        # 检索相关公式
        print("\n正在检索相关公式...")
        relevant_formulas = engine.retrieve_relevant_formulas(query, top_k=5)

        # 使用BGE rerank进行重排序
        print("正在进行重排序...")
        reranked_formulas = engine._rerank_with_bge(query, relevant_formulas)
        relevant_formulas = reranked_formulas[:3]  # 取前3个

        print(f"\n找到 {len(relevant_formulas)} 个相关公式:")
        for i, formula in enumerate(relevant_formulas, 1):
            print(f"{i}. {formula['name']} (相似度: {formula['similarity']:.4f})")
            print(f"   描述: {formula['description']}")

        # 选择第一个公式进行计算（实际应用中可能需要用户选择或进一步筛选）
        selected_formula = relevant_formulas[0]
        print(f"\n选择公式: {selected_formula['name']}")

        # 从问题中提取参数
        print("\n正在从问题中提取参数...")
        if use_qwen:
            parameters = engine.extract_parameters_with_qwen(query, selected_formula)
        else:
            # 使用原有的简单参数提取方法
            parameters = engine.extract_parameters_from_query(query, selected_formula)

        print(f"提取到的参数: {parameters}")

        # 检查是否有缺失的参数
        # missing_params = engine.get_missing_parameters(parameters)

        # if missing_params:
        #     print(f"\n缺少以下参数: {', '.join(missing_params)}")
        #     print("请重新提问并在问题中包含上述参数的值")
        #     continue

        # 执行计算
        print("\n正在执行计算...")
        result = await engine.execute_formula(selected_formula, parameters)

        # 显示结果
        if "error" in result:
            print(f"计算出错: {result['error']}")
        else:
            if result['unit']:
                print(f"\n计算结果: {result['result']} {result['unit']}")
            else:
                print(f"\n计算结果: {result['result']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

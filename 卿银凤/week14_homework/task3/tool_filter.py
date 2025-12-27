import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import mcp_tools


# 初始化RAG模型
class RAGToolSelector:
    def __init__(self, model_name: str="../Qwen/Qwen3-Embedding-0.6B"):
        # 使用Qwen模型
        self.model = SentenceTransformer(model_name)

        # 构建工具描述数据库
        self.tool_descriptions = self._build_tool_descriptions()
        self.tool_embeddings = self._encode_descriptions()

    def _build_tool_descriptions(self) -> List[Dict[str, Any]]:
        """构建工具描述数据库"""
        return [
            {
                "name": "get_bmr_rate",
                "description": """计算基础代谢率(BMR)，通过体重和身高估算每日基础代谢所需热量。
                公式：BMR = 10*weight + 6.25*height - 100""",
                "keywords": ["基础代谢率", "BMR", "代谢率", "热量", "体重", "身高", "能量需求", "健康评估"],
                "category": "医疗健康"
            },
            {
                "name": "get_property_value",
                "description": """估算房产市场价值，考虑面积、楼层系数和房龄因素。
                公式：value = 10000 * area * (1 + 0.02 * floor) * (1 - 0.015 * age)""",
                "keywords": ["房产价值", "房地产", "房价", "面积", "楼层", "房龄", "评估", "市场价值"],
                "category": "房地产"
            },
            {
                "name": "lake_pollutant_change_rate",
                "description": """计算湖泊污染物浓度变化率，考虑进水流量、浓度和自然降解。
                公式：dC/dt = (Qin*(Cin - C))/V - k*C""",
                "keywords": ["污染物", "浓度变化", "水质", "环境工程", "湖泊", "降解率", "流量"],
                "category": "环境工程"
            },
            {
                "name": "linear_model_output",
                "description": """线性模型预测，基于5个输入变量和权重系数。
                公式：y = 1.2*x1 - 0.5*x2 + 3.0*x3 + 2.1*x4 - 1.0*x5 + 4.5""",
                "keywords": ["线性模型", "预测", "线性回归", "变量", "权重", "系数", "趋势分析"],
                "category": "数据分析"
            },
            {
                "name": "combustion_efficiency",
                "description": """计算燃料燃烧效率，基于燃料热值。
                公式：η = 0.85 + 0.005*(HV - 40)，其中40 ≤ HV ≤ 50""",
                "keywords": ["燃烧效率", "燃料", "热值", "能源", "效率", "化工", "能源效率"],
                "category": "能源化工"
            },
            {
                "name": "fish_population_dynamics",
                "description": """预测鱼类种群数量变化，考虑自然增长、环境承载力和鱼苗投放。
                公式：P(t+1) = P(t) + r*P(t)*(1 - P(t)/K) + S""",
                "keywords": ["鱼类种群", "水产养殖", "种群动态", "增长率", "承载力", "鱼苗投放", "可持续发展"],
                "category": "水产养殖"
            },
            {
                "name": "soil_moisture_content",
                "description": """计算土壤含水量随深度变化的指数衰减关系。
                公式：θ(z) = θ0 * exp(-k * depth)""",
                "keywords": ["土壤含水量", "水分", "衰减", "深度", "农业", "灌溉", "土壤水分"],
                "category": "农业科学"
            },
            {
                "name": "texture_hardness_prediction",
                "description": """预测食品质地硬度，基于温度、时间、pH值和糖分含量。
                公式：hardness = 0.5*temp + 1.2*time - 3.0*ph + 0.8*sugar""",
                "keywords": ["质地硬度", "食品加工", "质量控制", "温度", "时间", "pH值", "糖分", "食品质量"],
                "category": "食品加工"
            },
            {
                "name": "environmental_quality_index",
                "description": """计算环境质量指数，考虑污染水平、人口密度和绿化覆盖率。
                公式：EQI = (100 - pollution_level) / (1 + population_density * (1 - green_coverage))""",
                "keywords": ["环境质量", "污染", "绿化", "人口密度", "环境指数", "宜居性", "环境评估"],
                "category": "环境评估"
            },
            {
                "name": "fish_daily_growth_random",
                "description": """预测鱼类日增重，考虑饲料、水温、密度和溶解氧，包含随机扰动。
                公式：ADG = 0.1*feed + 0.05*temp - 0.02*density + 0.03*oxygen + 随机扰动""",
                "keywords": ["日增重", "鱼类生长", "饲料", "水温", "养殖密度", "溶解氧", "水产养殖"],
                "category": "水产养殖"
            }
        ]

    def _encode_descriptions(self) -> np.ndarray:
        """编码所有工具描述"""
        descriptions = []
        for tool in self.tool_descriptions:
            # 组合描述和关键词
            text = tool["description"] + " " + " ".join(tool["keywords"])
            descriptions.append(text)

        # 编码所有描述
        return self.model.encode(descriptions)

    def rerank_tools(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """重排序工具，返回最相关的top_k个工具"""
        # 编码查询
        query_embedding = self.model.encode([query])

        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, self.tool_embeddings)[0]

        # 获取相似度最高的工具索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 构建结果
        results = []
        for idx in top_indices:
            tool_info = self.tool_descriptions[idx].copy()
            tool_info["similarity"] = float(similarities[idx])
            results.append(tool_info)

        return results

    def select_top_tool(self, query: str, threshold: float = 0.3) -> str:
        """选择最相关的工具，返回工具名称"""
        ranked_tools = self.rerank_tools(query, top_k=1)

        if ranked_tools and ranked_tools[0]["similarity"] >= threshold:
            return ranked_tools[0]["name"]
        else:
            return None


# MCP工具执行器
class MCPToolExecutor:
    def __init__(self):
        self.tools = {
            "get_bmr_rate": mcp_tools.get_bmr_rate,
            "get_property_value": mcp_tools.get_property_value,
            "lake_pollutant_change_rate": mcp_tools.lake_pollutant_change_rate,
            "linear_model_output": mcp_tools.linear_model_output,
            "combustion_efficiency": mcp_tools.combustion_efficiency,
            "fish_population_dynamics": mcp_tools.fish_population_dynamics,
            "soil_moisture_content": mcp_tools.soil_moisture_content,
            "texture_hardness_prediction": mcp_tools.texture_hardness_prediction,
            "environmental_quality_index": mcp_tools.environmental_quality_index,
            "fish_daily_growth_random": mcp_tools.fish_daily_growth_random
        }

    def extract_parameters(self, query: str, tool_name: str) -> List[float]:
        """从查询中提取数值参数"""
        # 使用正则表达式提取所有数字
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", query)

        # 转换为浮点数
        params = [float(num) for num in numbers]

        # 根据工具需求调整参数数量
        tool_param_counts = {
            "get_bmr_rate": 2,
            "get_property_value": 3,
            "lake_pollutant_change_rate": 5,
            "linear_model_output": 5,
            "combustion_efficiency": 1,
            "fish_population_dynamics": 4,
            "soil_moisture_content": 3,
            "texture_hardness_prediction": 4,
            "environmental_quality_index": 3,
            "fish_daily_growth_random": 4
        }

        required_count = tool_param_counts.get(tool_name, 0)

        if len(params) >= required_count:
            return params[:required_count]
        else:
            # 如果提取的参数不足，返回None列表
            return [None] * required_count

    def execute_tool(self, tool_name: str, params: List[float]) -> Any:
        """执行指定的MCP工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具 '{tool_name}' 不存在")

        # 检查是否有None参数
        if any(p is None for p in params):
            raise ValueError(f"参数提取不完整，无法执行 {tool_name}")

        # 调用对应的工具函数
        return self.tools[tool_name](*params)


# 问答系统主类
class MCPQASystem:
    def __init__(self):
        self.rag_selector = RAGToolSelector()
        self.tool_executor = MCPToolExecutor()

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """处理用户查询的完整流程"""
        result = {
            "query": user_query,
            "selected_tool": None,
            "similarity_score": None,
            "extracted_params": None,
            "tool_result": None,
            "explanation": None,
            "status": "success"
        }

        try:
            # 步骤1: 使用RAG选择工具
            selected_tool = self.rag_selector.select_top_tool(user_query)

            if not selected_tool:
                result["status"] = "error"
                result["explanation"] = "未找到与查询相关的工具"
                return result

            result["selected_tool"] = selected_tool

            # 获取相似度分数
            ranked = self.rag_selector.rerank_tools(user_query, top_k=1)
            if ranked:
                result["similarity_score"] = ranked[0]["similarity"]

            # 步骤2: 提取参数
            params = self.tool_executor.extract_parameters(user_query, selected_tool)
            result["extracted_params"] = params

            # 步骤3: 执行工具
            if any(p is None for p in params):
                result["status"] = "warning"
                result["explanation"] = "部分参数未提取到，使用默认值或需要进一步确认"
            else:
                tool_result = self.tool_executor.execute_tool(selected_tool, params)
                result["tool_result"] = tool_result

                # 根据工具类型生成解释
                result["explanation"] = self._generate_explanation(
                    selected_tool, params, tool_result
                )

        except Exception as e:
            result["status"] = "error"
            result["explanation"] = f"执行过程中出错: {str(e)}"

        return result

    def _generate_explanation(self, tool_name: str, params: List[float], result: Any) -> str:
        """根据工具结果生成解释文本"""
        explanations = {
            "get_bmr_rate": lambda p,
                                   r: f"根据您的体重{p[0]}kg和身高{p[1]}cm，您的基础代谢率(BMR)为{r}kcal/天。这表示您在静息状态下每天需要消耗的热量。",
            "get_property_value": lambda p,
                                         r: f"对于面积{p[0]}平方米、楼层系数{p[1]}、房龄{p[2]}年的房产，估算市场价值为{r}元。",
            "lake_pollutant_change_rate": lambda p,
                                                 r: f"在当前污染物浓度{p[0]}mg/L、进水流量{p[1]}m³/天、进水浓度{p[2]}mg/L、湖泊体积{p[3]}m³、降解速率{p[4]}/天的条件下，污染物浓度变化率为{r}mg/L/天。",
            "linear_model_output": lambda p,
                                          r: f"对于输入变量x1={p[0]}, x2={p[1]}, x3={p[2]}, x4={p[3]}, x5={p[4]}，线性模型预测输出为{r}。",
            "combustion_efficiency": lambda p, r: f"燃料热值为{p[0]}MJ/kg时，燃烧效率为{r * 100:.2f}%。",
            "fish_population_dynamics": lambda p,
                                               r: f"当前种群数量{p[0]}千尾，增长率{p[1]}，环境承载力{p[2]}千尾，新增投放{p[3]}千尾，下一时间步种群数量预测为{r}千尾。",
            "soil_moisture_content": lambda p, r: f"地表含水量{p[0]}，衰减系数{p[1]}，深度{p[2]}米处的土壤含水量为{r}。",
            "texture_hardness_prediction": lambda p,
                                                  r: f"在温度{p[0]}℃、时间{p[1]}分钟、pH值{p[2]}、糖分含量{p[3]}g/100g的工艺条件下，食品质地硬度预测为{r}。",
            "environmental_quality_index": lambda p,
                                                  r: f"污染水平{p[0]}、人口密度{p[1]}人/平方公里、绿化覆盖率{p[2]}的环境下，环境质量指数为{r}。",
            "fish_daily_growth_random": lambda p,
                                               r: f"饲料摄入量{p[0]}g/尾/天、水温{p[1]}℃、养殖密度{p[2]}尾/m³、溶解氧{p[3]}mg/L的条件下，鱼类日增重预测为{r}g/天。"
        }

        if tool_name in explanations:
            return explanations[tool_name](params, result)
        else:
            return f"工具'{tool_name}'的计算结果为: {result}"

    def format_response(self, result: Dict[str, Any]) -> str:
        """格式化输出结果"""
        if result["status"] == "error":
            return f"错误: {result['explanation']}"

        response = f"查询: {result['query']}\n"
        response += f"选择工具: {result['selected_tool']}\n"

        if result["similarity_score"]:
            response += f"相似度得分: {result['similarity_score']:.4f}\n"

        if result["extracted_params"]:
            response += f"提取参数: {result['extracted_params']}\n"

        if result["tool_result"] is not None:
            response += f"计算结果: {result['tool_result']}\n"

        if result["explanation"]:
            response += f"解释说明: {result['explanation']}\n"

        if result["status"] == "warning":
            response += "注意: 部分参数可能不完整，请确认参数值\n"

        return response


# 示例使用
if __name__ == "__main__":
    # 初始化问答系统
    qa_system = MCPQASystem()

    # 示例查询
    example_queries = [
        "体重70公斤身高175厘米的人基础代谢率是多少？",
        "100平方米的房子，楼层系数5，房龄10年，价值多少？",
        "湖泊污染物浓度5mg/L，进水流量1000m³/天，进水浓度8mg/L，体积50000m³，降解速率0.1/天，变化率是多少？",
        "线性模型输入1.2, 3.4, 5.6, 7.8, 9.0时输出多少？",
        "热值45MJ/kg的燃料燃烧效率是多少？"
    ]

    print("MCP工具问答系统演示\n" + "=" * 50)

    for query in example_queries:
        print(f"\n处理查询: {query}")
        result = qa_system.process_query(query)
        response = qa_system.format_response(result)
        print(response)
        print("-" * 50)

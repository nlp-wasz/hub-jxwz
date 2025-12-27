# import numpy as np
# import sympy as sp
from typing import Union, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
from task02_mcp_tools_from_pdf import *
import re
import json


class FormulaQASystem:
    def __init__(self):
        # 初始化语义相似度模型
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

        # 公式知识库：描述 -> (tool_function, 参数列表)
        self.formula_knowledge_base = self._build_knowledge_base()

        # 构建描述向量库
        self.descriptions = list(self.formula_knowledge_base.keys())
        self.description_embeddings = self.model.encode(self.descriptions,
                                                   convert_to_tensor=True)

    def _build_knowledge_base(self) -> Dict[str, Dict]:
        """构建公式知识库"""
        return {
            "用户数量指数增长模型，描述社交媒体内容传播的动态特性": {
                "tool": tool1,
                "params": ["t", "k", "u0"],
                "description": "用于预测用户关注度随时间指数增长的趋势"
            },
            "游客满意度动态评估模型，考虑温度人流设施交通因素": {
                "tool": tool2,
                "params": ["amenity_quality", "crowd_level", "travel_time", "temperature", "t"],
                "description": "量化环境和服务因素对游客满意度的影响"
            },
            "游客数量指数衰减模型，模拟闭园或非高峰期的客流变化": {
                "tool": tool3,
                "params": ["t", "k", "u0"],
                "description": "预测游客数量随时间衰减的趋势"
            },
            "结构挠度估算模型，建筑工程中梁板变形的刚度评估": {
                "tool": tool4,
                "params": ["load", "span", "stiffness"],
                "description": "计算简支梁在荷载作用下的最大挠度"
            },
            "旅游舒适度指数模型，基于气温湿度的环境舒适度评估": {
                "tool": tool5,
                "params": ["temp", "humidity"],
                "description": "综合气象条件评估旅游适宜性"
            },
            "电商转化率预测模型，多因素影响下的用户购买行为分析": {
                "tool": tool6,
                "params": ["price", "discount", "time", "stock", "rating"],
                "description": "预测商品页面的用户转化概率"
            },
            "混凝土用量估算模型，建筑工程材料计划与成本控制": {
                "tool": tool7,
                "params": ["length", "width", "height"],
                "description": "考虑施工损耗的混凝土体积计算"
            },
            "车辆油耗预测模型，非线性回归分析驾驶能耗": {
                "tool": tool8,
                "params": ["speed", "acceleration", "engine_load", "temperature", "road_gradient"],
                "description": "预测不同工况下的百公里油耗"
            },
            "食品成分降解动力学模型，热处理过程中的营养损失预测": {
                "tool": tool9,
                "params": ["t", "C0", "k", "Ea", "T"],
                "description": "基于Arrhenius方程的浓度变化模拟"
            },
            "种群动态演化模型，空间生态学中的反应扩散方程": {
                "tool": tool10,
                "params": ["x", "t", "u0", "D", "r", "K", "dx", "dt"],
                "description": "一维空间中种群密度随时间空间的变化"
            }
        }

    def rag_tool_selection(self, user_question: str, top_k: int = 3) -> List[Dict]:
        """
        步骤3: RAG工具选择
        使用语义相似度匹配用户问题与公式描述
        """
        # 编码用户问题
        question_embedding = self.model.encode([user_question], convert_to_tensor=True)

        # 计算相似度
        similarities = util.cos_sim(question_embedding, self.description_embeddings)[0]

        # 获取top-k结果
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            desc = self.descriptions[idx]
            tool_info = self.formula_knowledge_base[desc]
            results.append({
                "description": desc,
                "similarity": float(similarities[idx]),
                "tool_info": tool_info
            })

        return results

    def extract_parameters(self, user_question: str, tool_params: List[str]) -> Dict[str, Any]:
        """使用LLM智能提取参数"""

        prompt = f"""
        请从用户问题中提取以下参数：{tool_params}

        用户问题：{user_question}

        请按照JSON格式返回提取到的参数，只包含有明确数值的参数。
        示例输出：{{"t": 5.0, "k": 0.2, "u0": 1000}}

        如果没有找到某个参数的值，请不要包含该参数。
        返回结果：
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
            return {}

        except Exception as e:
            print(f"LLM参数提取失败: {e}")
            return self.fallback_extraction(user_question, tool_params)

    def fallback_extraction(self, user_question: str, tool_params: List[str]) -> Dict[str, Any]:
        """备用方案：基于模式的参数提取"""
        params = {}
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', user_question)

        # 简单映射：按出现顺序分配参数
        for i, param in enumerate(tool_params):
            if i < len(numbers):
                params[param] = float(numbers[i])

        return params

    def execute_tool(self, tool_function, params: Dict) -> Dict[str, Any]:
        """
        步骤4: 执行选定的工具
        """
        try:
            # 参数验证
            missing_params = []
            for param in tool_function.__code__.co_varnames[:tool_function.__code__.co_argcount]:
                if param not in params and param != 'self':
                    missing_params.append(param)

            if missing_params:
                return {
                    "success": False,
                    "error": f"缺少参数: {missing_params}",
                    "suggested_params": missing_params
                }

            # 调用工具函数
            result = tool_function(**params)

            return {
                "success": True,
                "result": result,
                "used_params": params
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "used_params": params
            }

    def ask_question(self, user_question: str, manual_params: Dict = None) -> Dict[str, Any]:
        """
        智能问答主入口
        """
        # 步骤3: RAG工具选择
        tool_candidates = self.rag_tool_selection(user_question, top_k=3)

        if not tool_candidates:
            return {"error": "未找到匹配的公式模型"}

        # 选择相似度最高的工具
        selected_tool = tool_candidates[0]

        # 参数提取
        if manual_params:
            params = manual_params
        else:
            params = self.extract_parameters(user_question, selected_tool["tool_info"])

        # 步骤4: 执行计算
        execution_result = self.execute_tool(
            selected_tool["tool_info"]["tool"],
            params
        )

        # 汇总回答
        response = {
            "question": user_question,
            "selected_tool": selected_tool["description"],
            "similarity_score": selected_tool["similarity"],
            "execution_result": execution_result,
            "alternative_tools": [
                {
                    "description": tool["description"],
                    "similarity": tool["similarity"]
                }
                for tool in tool_candidates[1:]  # 排除已选工具
            ]
        }

        return response

    def batch_process(self, questions: List[str]) -> List[Dict]:
        """批量处理多个问题"""
        results = []
        for question in questions:
            result = self.ask_question(question)
            results.append(result)
        return results


# MCP服务器封装
class FormulaMCPServer:
    def __init__(self):
        self.qas_system = FormulaQASystem()

    def handle_request(self, request_data: Dict) -> Dict:
        """处理MCP请求"""
        question = request_data.get("question", "")
        params = request_data.get("parameters", {})

        if not question:
            return {"error": "问题不能为空"}

        response = self.qas_system.ask_question(question, params)
        return response

    def get_available_tools(self) -> List[Dict]:
        """获取可用的工具列表"""
        tools = []
        for desc, info in self.qas_system.formula_knowledge_base.items():
            tools.append({
                "name": info["tool"].__name__,
                "description": desc,
                "parameters": info["params"],
                "detailed_description": info["description"]
            })
        return tools


# 示例使用
if __name__ == "__main__":
    # 初始化系统
    mcp_server = FormulaMCPServer()
    qas_system = FormulaQASystem()

    # 测试示例
    test_questions = [
        "预测社交媒体内容在5小时后的用户数量，初始用户1000人，增长速率0.2",
        "计算温度为25度，湿度60%时的旅游舒适度",
        "估算长度为10米、宽度5米、高度3米的混凝土用量",
        "预测车辆在速度80km/h，加速度2m/s²时的油耗"
    ]

    print("=== 公式智能问答系统测试 ===")

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 问题 {i}: {question} ---")
        result = qas_system.ask_question(question)

        if "execution_result" in result and result["execution_result"]["success"]:
            print(f"匹配工具: {result['selected_tool']}")
            print(f"相似度: {result['similarity_score']:.3f}")
            print(f"计算结果: {result['execution_result']['result']}")
        else:
            print(f"错误: {result.get('error', '未知错误')}")

    # 显示可用工具
    print(f"\n=== 系统支持 {len(mcp_server.get_available_tools())} 个公式工具 ===")
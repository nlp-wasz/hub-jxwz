import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


# ==================== 工具定义区域（从文档中复制） ====================

@dataclass
class MCPTool:
    name: str
    description: str
    parameters: List[str]
    function: callable


class ToolType(Enum):
    TEMPORAL_SIMULATOR = "temporal_simulator"
    DAILY_ORDER_PREDICTOR = "daily_order_predictor"
    YIELD_PREDICTOR = "yield_predictor"
    FOOD_DRYING = "food_drying_evaporation_calculator"
    STUDENT_PERFORMANCE = "student_performance_predictor"
    DYNAMIC_SYSTEM = "dynamic_system_predictor"
    QUADRATIC_RESPONSE = "quadratic_response_model"
    MEDIA_INFLUENCE = "media_influence_evaluator"
    CATTLE_POPULATION = "cattle_population_next_year"
    WEIGHT_PREDICTOR = "weight_predictor"


# ==================== 工具实现 ====================

def temporal_simulator(t: float, a: float, b: float, c: float, d: float) -> float:
    """计算时间 t 时的溶解氧浓度。"""
    exponential_term = a * math.exp(-b * t)
    sinusoidal_term = c * math.sin(d * t)
    concentration = exponential_term + sinusoidal_term
    return concentration


def daily_order_predictor(ad_spend: float, discount_rate: float, prev_orders: float) -> float:
    """预测电商当日订单量。"""
    alpha = 0.05
    beta = 100.0
    gamma = 0.7

    orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
    return orders


def yield_predictor(F: float, I: float, T: float, a: float, b: float, c: float) -> float:
    """预测作物单位面积产量。"""
    Y = a * F + b * I - c * (T ** 2)
    return Y


def food_drying_evaporation_calculator(M0: float, k: float, T: float) -> float:
    """计算累计水分蒸发量: Evap(T) = M0 * (T + (exp(-k*T) - 1) / k)"""
    if k <= 0 or M0 < 0 or T < 0:
        raise ValueError("参数必须满足 M0 >= 0, k > 0, T >= 0")

    exp_term = math.exp(-k * T)
    evaporated = M0 * (T + (exp_term - 1) / k)
    return evaporated


def student_performance_predictor(
        study_hours: float,
        attendance_rate: float,
        quiz_avg_score: float,
        participation_score: float,
        weights: list,
        alpha: float,
        beta: float
) -> float:
    """预测学生课程成绩： mapped_participation = (participation_score - 1) * 25"""
    w1, w2, w3, w4 = weights
    participation_mapped = (participation_score - 1) * 25.0
    linear_comb = (
            w1 * study_hours +
            w2 * attendance_rate +
            w3 * quiz_avg_score +
            w4 * participation_mapped
    )
    exponent = -alpha * (linear_comb - beta)
    score = 100.0 / (1.0 + math.exp(exponent))
    return score


def dynamic_system_predictor(
        x1: float, x2: float, x3: float,
        y_prev1: float, y_prev2: float,
        a: float, b: float, c: float, d: float
) -> float:
    """预测动态系统的当前输出。"""
    y_t = a * x1 + b * y_prev1 + c * y_prev2 + d * x2 * x3
    return y_t


def quadratic_response_model(x: float) -> float:
    """计算二次函数 y = 2x^2 + 3x + 1 的输出。"""
    return 2 * x ** 2 + 3 * x + 1


def media_influence_evaluator(
        content_quality: float,
        channels: int,
        engagement: float,
        time: float
) -> float:
    """计算传播项目的综合影响力"""
    if content_quality < 0 or engagement < 0 or time < 0 or channels < 1:
        raise ValueError("输入参数必须满足非负约束，且 channels >= 1")

    influence = content_quality * channels * engagement * time
    return influence


def cattle_population_next_year(N_current: float, r: float, K: float) -> float:
    """预测下一年牛群数量"""
    if N_current < 0 or r < 0 or K <= 0:
        raise ValueError("N_current >= 0, r >= 0, K > 0 required")

    N_next = N_current + r * N_current * (1 - N_current / K)
    return N_next


def weight_predictor(W0: float, r: float, t: float) -> float:
    """预测牲畜在时间 t 的体重"""
    if W0 <= 0 or r < 0 or t < 0:
        raise ValueError("W0 > 0, r >= 0, t >= 0 required")

    W_t = W0 * math.exp(r * t)
    return W_t


# ==================== 工具知识库 ====================

TOOLS_KNOWLEDGE_BASE = {
    ToolType.TEMPORAL_SIMULATOR: {
        "name": "temporal_simulator",
        "description": "计算时间 t 时的溶解氧浓度，基于指数衰减和正弦扰动模型",
        "keywords": ["溶解氧", "氧气浓度", "时间序列", "衰减", "正弦", "水质", "环境监测"],
        "parameters": ["t: 时间", "a: 初始DO释放量", "b: 衰减系数", "c: 扰动振幅", "d: 扰动频率"],
        "function": temporal_simulator
    },
    ToolType.DAILY_ORDER_PREDICTOR: {
        "name": "daily_order_predictor",
        "description": "预测电商当日订单量，基于广告支出、折扣率和历史订单",
        "keywords": ["订单预测", "电商", "广告", "折扣", "销售预测", "零售"],
        "parameters": ["ad_spend: 广告支出", "discount_rate: 折扣率", "prev_orders: 历史订单数"],
        "function": daily_order_predictor
    },
    ToolType.YIELD_PREDICTOR: {
        "name": "yield_predictor",
        "description": "预测作物单位面积产量，基于土壤肥力、灌溉量和气温",
        "keywords": ["作物产量", "农业", "土壤肥力", "灌溉", "气温", "农场"],
        "parameters": ["F: 土壤肥力指数", "I: 每周灌溉量", "T: 平均气温", "a,b,c: 经验系数"],
        "function": yield_predictor
    },
    ToolType.FOOD_DRYING: {
        "name": "food_drying_evaporation_calculator",
        "description": "计算食品干燥过程中的累计水分蒸发量",
        "keywords": ["食品干燥", "水分蒸发", "食品加工", "脱水", "干燥过程"],
        "parameters": ["M0: 初始水分含量", "k: 干燥速率常数", "T: 干燥时间"],
        "function": food_drying_evaporation_calculator
    },
    ToolType.STUDENT_PERFORMANCE: {
        "name": "student_performance_predictor",
        "description": "预测学生课程成绩，基于学习时长、出勤率、测验成绩和参与度",
        "keywords": ["学生成绩", "教育", "学习分析", "成绩预测", "教学评估"],
        "parameters": ["study_hours: 学习时长", "attendance_rate: 出勤率", "quiz_avg_score: 测验平均分",
                       "participation_score: 参与度分数", "weights: 权重列表", "alpha: 参数", "beta: 参数"],
        "function": student_performance_predictor
    },
    ToolType.DYNAMIC_SYSTEM: {
        "name": "dynamic_system_predictor",
        "description": "预测动态系统的当前输出，基于当前输入和历史输出",
        "keywords": ["动态系统", "系统预测", "控制理论", "状态空间", "系统建模"],
        "parameters": ["x1,x2,x3: 当前输入", "y_prev1,y_prev2: 历史输出", "a,b,c,d: 系统参数"],
        "function": dynamic_system_predictor
    },
    ToolType.QUADRATIC_RESPONSE: {
        "name": "quadratic_response_model",
        "description": "计算二次函数的输出值",
        "keywords": ["二次函数", "多项式", "函数", "曲线拟合", "回归"],
        "parameters": ["x: 输入变量"],
        "function": quadratic_response_model
    },
    ToolType.MEDIA_INFLUENCE: {
        "name": "media_influence_evaluator",
        "description": "计算传播项目的综合影响力",
        "keywords": ["媒体影响力", "传播效果", "内容质量", "渠道", "参与度"],
        "parameters": ["content_quality: 内容质量", "channels: 渠道数量", "engagement: 参与度", "time: 时间"],
        "function": media_influence_evaluator
    },
    ToolType.CATTLE_POPULATION: {
        "name": "cattle_population_next_year",
        "description": "预测下一年牛群数量，基于逻辑增长模型",
        "keywords": ["牛群数量", "畜牧业", "种群增长", "养殖", "动物数量"],
        "parameters": ["N_current: 当前数量", "r: 增长率", "K: 环境容量"],
        "function": cattle_population_next_year
    },
    ToolType.WEIGHT_PREDICTOR: {
        "name": "weight_predictor",
        "description": "预测牲畜在时间 t 的体重，基于指数增长模型",
        "keywords": ["牲畜体重", "生长模型", "动物生长", "体重预测", "养殖"],
        "parameters": ["W0: 初始体重", "r: 生长率", "t: 时间"],
        "function": weight_predictor
    }
}


# ==================== RAG工具选择器 ====================

class ToolSelector:
    def __init__(self, tools_kb: Dict[ToolType, Dict]):
        self.tools_kb = tools_kb
        self.all_keywords = self._extract_all_keywords()

    def _extract_all_keywords(self) -> List[str]:
        """提取所有工具的关键词"""
        keywords = []
        for tool_info in self.tools_kb.values():
            keywords.extend(tool_info["keywords"])
        return list(set(keywords))

    def calculate_similarity(self, query: str, tool_info: Dict) -> float:
        """计算查询与工具描述的相似度"""
        query_lower = query.lower()
        tool_text = f"{tool_info['description']} {' '.join(tool_info['keywords'])}".lower()

        # 简单关键词匹配相似度计算
        score = 0
        for keyword in tool_info["keywords"]:
            if keyword.lower() in query_lower:
                score += 2

        # 描述匹配
        for word in tool_info["description"].lower().split():
            if word in query_lower and len(word) > 3:
                score += 1

        return score

    def select_tool(self, query: str) -> Optional[Tuple[ToolType, Dict]]:
        """根据用户查询选择最相关的工具"""
        best_tool = None
        best_score = -1

        for tool_type, tool_info in self.tools_kb.items():
            score = self.calculate_similarity(query, tool_info)
            if score > best_score:
                best_score = score
                best_tool = (tool_type, tool_info)

        # 设置阈值，避免低相似度匹配
        if best_score > 0:
            return best_tool
        return None


# ==================== 参数提取器 ====================

class ParameterExtractor:
    @staticmethod
    def extract_parameters(query: str, tool_type: ToolType) -> Dict[str, Any]:
        if tool_type == ToolType.TEMPORAL_SIMULATOR:
            return {
                "t": 5.0,  # 默认值，实际应从查询中提取
                "a": 10.0,
                "b": 0.1,
                "c": 2.0,
                "d": 0.5
            }
        elif tool_type == ToolType.DAILY_ORDER_PREDICTOR:
            return {
                "ad_spend": 1000.0,
                "discount_rate": 0.1,
                "prev_orders": 500.0
            }
        elif tool_type == ToolType.YIELD_PREDICTOR:
            return {
                "F": 80.0,
                "I": 50.0,
                "T": 25.0,
                "a": 0.5,
                "b": 0.3,
                "c": 0.01
            }
        elif tool_type == ToolType.FOOD_DRYING:
            return {
                "M0": 100.0,
                "k": 0.1,
                "T": 10.0
            }
        elif tool_type == ToolType.STUDENT_PERFORMANCE:
            return {
                "study_hours": 20.0,
                "attendance_rate": 0.9,
                "quiz_avg_score": 85.0,
                "participation_score": 3.0,
                "weights": [0.3, 0.2, 0.3, 0.2],
                "alpha": 0.1,
                "beta": 50.0
            }
        elif tool_type == ToolType.DYNAMIC_SYSTEM:
            return {
                "x1": 1.0,
                "x2": 2.0,
                "x3": 3.0,
                "y_prev1": 10.0,
                "y_prev2": 8.0,
                "a": 0.5,
                "b": 0.3,
                "c": 0.2,
                "d": 0.1
            }
        elif tool_type == ToolType.QUADRATIC_RESPONSE:
            return {"x": 2.0}
        elif tool_type == ToolType.MEDIA_INFLUENCE:
            return {
                "content_quality": 0.8,
                "channels": 5,
                "engagement": 0.7,
                "time": 30.0
            }
        elif tool_type == ToolType.CATTLE_POPULATION:
            return {
                "N_current": 100.0,
                "r": 0.15,
                "K": 500.0
            }
        elif tool_type == ToolType.WEIGHT_PREDICTOR:
            return {
                "W0": 200.0,
                "r": 0.02,
                "t": 365.0
            }

        return {}


# ==================== MCP工具执行器 ====================

class MCPExecutor:
    def __init__(self):
        self.tool_selector = ToolSelector(TOOLS_KNOWLEDGE_BASE)
        self.param_extractor = ParameterExtractor()

    def execute_query(self, user_query: str) -> Dict[str, Any]:
        """执行完整的RAG流程：选择工具 -> 提取参数 -> 执行计算 -> 返回结果"""

        # 步骤1: RAG工具选择
        selected = self.tool_selector.select_tool(user_query)

        if not selected:
            return {
                "success": False,
                "error": "未找到与查询匹配的工具",
                "suggested_tools": list(TOOLS_KNOWLEDGE_BASE.keys())
            }

        tool_type, tool_info = selected

        # 步骤2: 提取参数
        try:
            params = self.param_extractor.extract_parameters(user_query, tool_type)

            # 步骤3: 执行工具
            result = tool_info["function"](**params)

            return {
                "success": True,
                "tool_selected": tool_info["name"],
                "tool_description": tool_info["description"],
                "parameters_used": params,
                "result": result,
                "execution_details": f"成功执行 {tool_info['name']}，计算结果为 {result:.4f}"
            }

        except ValueError as e:
            return {
                "success": False,
                "error": f"参数错误: {str(e)}",
                "tool_selected": tool_info["name"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"执行错误: {str(e)}",
                "tool_selected": tool_info["name"]
            }


def main():
    print("可用工具领域：")
    print("1. 环境监测（溶解氧浓度）")
    print("2. 电商预测（订单量预测）")
    print("3. 农业预测（作物产量）")
    print("4. 食品加工（水分蒸发）")
    print("5. 教育评估（学生成绩）")
    print("6. 系统控制（动态系统）")
    print("7. 数学函数（二次函数）")
    print("8. 媒体分析（影响力评估）")
    print("9. 畜牧业（种群预测）")
    print("10. 畜牧业（体重预测）")

    executor = MCPExecutor()

    while True:
        print("\n请输入您的问题或需求（输入'退出'结束）：")
        user_input = input("> ")

        if user_input.lower() in ['退出', 'exit', 'quit']:
            break

        if not user_input.strip():
            continue

        # 执行查询
        result = executor.execute_query(user_input)

        # 显示结果
        print("执行结果：")

        if result["success"]:
            print(f"✓ 已选择工具：{result['tool_selected']}")
            print(f"  工具描述：{result['tool_description']}")
            print(f"  使用参数：{result['parameters_used']}")
            print(f"  计算结果：{result['result']:.4f}")
        else:
            print(f"✗ 执行失败：{result['error']}")
            if 'suggested_tools' in result:
                print("  可用工具列表：")
                for tool in result['suggested_tools']:
                    print(f"  - {tool.value}")



if __name__ == "__main__":
    main()

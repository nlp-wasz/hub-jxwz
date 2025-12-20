import numpy as np
import math
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Complex System Nonlinear Interaction Model")

@mcp.tool()
def calculate_nonlinear_interaction(
    x: float = 1.0,        # 输入变量 x
    y: float = 1.0         # 输入变量 y
) -> dict:
    """
    复杂系统非线性交互作用模型
    
    建模背景:
    在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。
    该模型旨在模拟两个输入变量 x 和 y 对某一目标输出的综合影响，其中包含了
    周期性变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、
    多因子耦合作用下的信号响应机制等场景。尽管模型本身为确定性函数，但其结构设计
    使得输出呈现出类随机波动的特性，从而更好地模拟真实世界中的复杂行为。
    
    建模公式:
    非线性交互作用函数如下：
    
    \[f_{un}(x, y) = 2.5 \cdot \sin(x) + 1.8 \cdot \cos(y) + 0.3 \cdot x \cdot y\]
    
    该公式融合了变量 x 和 y 的非线性主效应及其乘积项所代表的交互效应，能够
    有效刻画输入变量之间的动态相互作用。
    
    参数说明:
    - x: 输入变量 x，可以是任意实数
    - y: 输入变量 y，可以是任意实数
    
    返回:
    - dict: 包含函数值、各分量贡献和模型参数的字典
    
    使用示例:
    >>> result = calculate_nonlinear_interaction(x=1.0, y=1.0)
    >>> print(f"函数值: {result['function_value']:.4f}")
    """
    
    # 1. 计算函数值：f(x,y) = 2.5*sin(x) + 1.8*cos(y) + 0.3*x*y
    sin_term = 2.5 * math.sin(x)
    cos_term = 1.8 * math.cos(y)
    interaction_term = 0.3 * x * y
    
    function_value = sin_term + cos_term + interaction_term
    
    # 2. 计算各分量的贡献百分比
    total_abs = abs(sin_term) + abs(cos_term) + abs(interaction_term)
    
    if total_abs > 0:
        sin_contribution = abs(sin_term) / total_abs * 100
        cos_contribution = abs(cos_term) / total_abs * 100
        interaction_contribution = abs(interaction_term) / total_abs * 100
    else:
        sin_contribution = cos_contribution = interaction_contribution = 0.0
    
    # 3. 识别主导效应
    max_contribution = max(sin_contribution, cos_contribution, interaction_contribution)
    if max_contribution == sin_contribution:
        dominant_effect = "正弦项主导"
    elif max_contribution == cos_contribution:
        dominant_effect = "余弦项主导"
    else:
        dominant_effect = "交互项主导"
    
    # 4. 计算函数性质
    # 周期性分析
    sin_period = 2 * math.pi  # 正弦项周期
    cos_period = 2 * math.pi  # 余弦项周期
    
    # 交互项的线性度
    interaction_linearity = "线性" if x != 0 and y != 0 else "零交互"
    
    # 5. 评估函数输出范围（近似估计）
    # 正弦项范围：[-2.5, 2.5]
    # 余弦项范围：[-1.8, 1.8]
    # 交互项范围：取决于x和y的范围
    estimated_min = -2.5 - 1.8 - abs(0.3 * x * y)
    estimated_max = 2.5 + 1.8 + abs(0.3 * x * y)
    
    # 6. 生成模型分析
    model_analysis = []
    
    if abs(sin_term) > abs(cos_term) + abs(interaction_term):
        model_analysis.append("正弦效应对输出影响最大")
    elif abs(cos_term) > abs(sin_term) + abs(interaction_term):
        model_analysis.append("余弦效应对输出影响最大")
    elif abs(interaction_term) > abs(sin_term) + abs(cos_term):
        model_analysis.append("交互效应对输出影响最大")
    
    if interaction_term > 0:
        model_analysis.append("x和y之间存在正交互作用")
    elif interaction_term < 0:
        model_analysis.append("x和y之间存在负交互作用")
    else:
        model_analysis.append("x和y之间无交互作用")
    
    # 判断输出是否在典型范围内
    if function_value > 3.0:
        model_analysis.append("输出值较高")
    elif function_value < -3.0:
        model_analysis.append("输出值较低")
    else:
        model_analysis.append("输出值在中等范围")
    
    return {
        "function_value": float(function_value),
        "component_values": {
            "sin_term": float(sin_term),
            "cos_term": float(cos_term),
            "interaction_term": float(interaction_term)
        },
        "component_contributions": {
            "sin_contribution_percent": float(sin_contribution),
            "cos_contribution_percent": float(cos_contribution),
            "interaction_contribution_percent": float(interaction_contribution),
            "dominant_effect": dominant_effect
        },
        "function_properties": {
            "estimated_min_value": float(estimated_min),
            "estimated_max_value": float(estimated_max),
            "sin_period": float(sin_period),
            "cos_period": float(cos_period),
            "interaction_linearity": interaction_linearity
        },
        "model_parameters": {
            "x": float(x),
            "y": float(y)
        },
        "model_analysis": model_analysis,
        "formula_explanation": {
            "formula": "f(x,y) = 2.5 * sin(x) + 1.8 * cos(y) + 0.3 * x * y",
            "terms": {
                "2.5 * sin(x)": "x的正弦项，表示x的周期性主效应",
                "1.8 * cos(y)": "y的余弦项，表示y的周期性主效应", 
                "0.3 * x * y": "交互项，表示x和y的耦合作用"
            }
        }
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
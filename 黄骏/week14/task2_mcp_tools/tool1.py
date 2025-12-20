import numpy as np
import math
from scipy import integrate
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Complex System Multi-Variable Nonlinear Model")

@mcp.tool()
def calculate_complex_system(
    x1: float = 1.0,
    x2: float = 1.0,
    x3: float = 0.0,
    x4: float = 1.0,
    x5: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    e: float = 1.0,
    s: float = 1.0
) -> dict:
    """
    复杂系统多变量非线性模型
    
    建模背景:
    本模型设计用于描述一个具有多变量输入的复杂系统，其输出由多种非线性机制共同决定。
    系统中包含随时间或空间累积的动态效应，通过积分形式进行建模，能够有效刻画具有衰减特性的累积过程。
    此外，模型还综合考虑了输入变量的二次效应、周期性响应、对数变换和幂律关系，
    从而增强对实际物理、工程或金融现象的拟合能力和解释性。
    
    建模公式:
    模型输出由积分项与非线性项的叠加构成，表达式为：
    
    \[y = \int_{0}^{s \cdot x_1} \frac{a \cdot e^{-t/b}}{c + t} dt + d \cdot x_2^2 + e \cdot \sin(x_3) + \log(x_4 + 1) + \sqrt{x_5}\]
    
    其中各项分别对应累积衰减过程、二次响应、周期性响应、对数变换和幂律变换。
    模型参数 \(s, a, b, c, d, e\) 易于调节各部分的强度和动态范围，
    积分上限 \(s \times x_1\) 与其他输入变量共同决定了系统的整体响应值。
    
    参数说明:
    - x1: 输入变量1，用于确定积分上限的变量
    - x2: 输入变量2，用于二次响应项
    - x3: 输入变量3，用于周期性响应项
    - x4: 输入变量4，用于对数变换项
    - x5: 输入变量5，用于幂律变换项
    - a: 积分项强度系数
    - b: 积分项衰减系数
    - c: 积分项偏移系数
    - d: 二次项系数
    - e: 正弦项系数
    - s: 积分上限调节系数
    
    返回:
    - dict: 包含输出值、各分量值和模型参数的字典
    
    使用示例:
    >>> result = calculate_complex_system(
    ...     x1=1.0, x2=1.0, x3=0.0, x4=1.0, x5=1.0,
    ...     a=1.0, b=1.0, c=1.0, d=1.0, e=1.0, s=1.0
    ... )
    >>> print(f"系统输出值: {result['output_value']:.4f}")
    """
    
    # 1. 参数验证
    # 确保对数参数为正
    if x4 + 1 <= 0:
        raise ValueError("x4必须大于-1以确保对数参数为正")
    
    # 确保平方根参数非负
    if x5 < 0:
        raise ValueError("x5必须为非负数以确保平方根有意义")
    
    # 2. 计算积分项
    # 积分上限
    upper_limit = s * x1
    
    if upper_limit <= 0:
        integral_term = 0.0
    else:
        # 定义被积函数
        def integrand(t):
            denominator = c + t
            # 避免分母为零
            if abs(denominator) < 1e-12:
                denominator = 1e-12
            return a * math.exp(-t / b) / denominator
        
        # 计算定积分
        integral_term, error = integrate.quad(integrand, 0, upper_limit)
    
    # 3. 计算二次项
    quadratic_term = d * (x2 ** 2)
    
    # 4. 计算正弦项
    sin_term = e * math.sin(x3)
    
    # 5. 计算对数项
    log_term = math.log(x4 + 1)
    
    # 6. 计算平方根项
    sqrt_term = math.sqrt(x5)
    
    # 7. 计算总输出
    output_value = integral_term + quadratic_term + sin_term + log_term + sqrt_term
    
    # 8. 计算各分量的贡献百分比（绝对值贡献）
    abs_terms = [
        abs(integral_term),
        abs(quadratic_term),
        abs(sin_term),
        abs(log_term),
        abs(sqrt_term)
    ]
    total_abs = sum(abs_terms)
    
    if total_abs > 0:
        integral_contribution = abs_terms[0] / total_abs * 100
        quadratic_contribution = abs_terms[1] / total_abs * 100
        sin_contribution = abs_terms[2] / total_abs * 100
        log_contribution = abs_terms[3] / total_abs * 100
        sqrt_contribution = abs_terms[4] / total_abs * 100
    else:
        integral_contribution = quadratic_contribution = sin_contribution = log_contribution = sqrt_contribution = 0.0
    
    # 9. 识别主导分量
    contributions = {
        "积分项": integral_contribution,
        "二次项": quadratic_contribution,
        "正弦项": sin_contribution,
        "对数项": log_contribution,
        "平方根项": sqrt_contribution
    }
    max_component = max(contributions, key=contributions.get)
    max_contribution = contributions[max_component]
    
    # 10. 分析各项特性
    # 积分项特性
    if integral_term != 0:
        if upper_limit > 0:
            integrand_at_end = a * math.exp(-upper_limit / b) / (c + upper_limit)
            integral_character = f"积分上限{s*x1:.2f}，被积函数末端值{integrand_at_end:.4f}"
        else:
            integral_character = "积分上限为0，无积分效应"
    else:
        integral_character = "积分项为0"
    
    # 正弦项特性
    sin_amplitude = abs(e)
    sin_period = 2 * math.pi
    sin_phase = x3 % (2 * math.pi)
    
    # 11. 生成模型分析
    model_analysis = []
    
    if integral_contribution > 50:
        model_analysis.append("系统行为主要由累积衰减过程主导")
    elif quadratic_contribution > 50:
        model_analysis.append("系统行为主要由二次响应主导")
    elif sin_contribution > 50:
        model_analysis.append("系统行为主要由周期性响应主导")
    
    if sin_term < 0:
        model_analysis.append("正弦项当前为负相位")
    elif sin_term > 0:
        model_analysis.append("正弦项当前为正相位")
    
    if quadratic_term > 10:
        model_analysis.append("二次项效应较强")
    
    if log_term > 2:
        model_analysis.append("对数变换效应明显")
    
    if sqrt_term > 3:
        model_analysis.append("幂律变换效应明显")
    
    # 判断输出值范围
    if output_value > 100:
        output_range = "极高"
    elif output_value > 10:
        output_range = "高"
    elif output_value > 1:
        output_range = "中等"
    elif output_value > 0:
        output_range = "低"
    elif output_value > -1:
        output_range = "负低值"
    else:
        output_range = "负高值"
    
    return {
        "output_value": float(output_value),
        "component_values": {
            "integral_term": float(integral_term),
            "quadratic_term": float(quadratic_term),
            "sin_term": float(sin_term),
            "log_term": float(log_term),
            "sqrt_term": float(sqrt_term)
        },
        "component_contributions": {
            "integral_contribution_percent": float(integral_contribution),
            "quadratic_contribution_percent": float(quadratic_contribution),
            "sin_contribution_percent": float(sin_contribution),
            "log_contribution_percent": float(log_contribution),
            "sqrt_contribution_percent": float(sqrt_contribution),
            "dominant_component": max_component,
            "dominant_contribution": float(max_contribution)
        },
        "component_characteristics": {
            "integral": integral_character,
            "sin_amplitude": float(sin_amplitude),
            "sin_period": float(sin_period),
            "sin_phase": float(sin_phase)
        },
        "model_parameters": {
            "x1": float(x1),
            "x2": float(x2),
            "x3": float(x3),
            "x4": float(x4),
            "x5": float(x5),
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "e": float(e),
            "s": float(s)
        },
        "model_analysis": {
            "analysis_items": model_analysis,
            "output_range": output_range,
            "upper_limit": float(upper_limit)
        },
        "formula_explanation": {
            "full_formula": "y = ∫₀^{s·x₁} [a·exp(-t/b)/(c+t)] dt + d·x₂² + e·sin(x₃) + log(x₄+1) + √x₅",
            "terms": {
                "integral": "累积衰减过程项，表示随时间衰减的累积效应",
                "quadratic": "二次响应项，表示x₂的二次方效应",
                "sin": "周期性响应项，表示x₃的正弦周期性效应",
                "log": "对数变换项，表示x₄的对数增长效应",
                "sqrt": "幂律变换项，表示x₅的平方根效应"
            }
        }
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
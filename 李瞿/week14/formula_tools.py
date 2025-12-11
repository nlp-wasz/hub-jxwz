"""
定义10个基于文档公式的MCP工具
每个工具对应一个具体的数学公式计算功能
"""

import numpy as np
import sympy as sp
from sympy import symbols, exp, log, sqrt, Abs, integrate, Symbol
from fastmcp import FastMCP

# 初始化FastMCP实例
mcp = FastMCP("Formula QA System")


@mcp.tool()
def agricultural_yield_prediction(
    temp: float,
    rainfall: float,
    fertilizer: float,
    sunlight: float,
    soil_quality: float
) -> float:
    """
    农业产量预测模型
    基于温度、降水量、施肥量、光照时长和土壤质量指数预测作物产量
    
    参数:
    - temp: 平均生长温度(℃)
    - rainfall: 生长期间降水量(mm)
    - fertilizer: 施肥量
    - sunlight: 每日平均光照时长(小时)
    - soil_quality: 土壤质量指数
    
    返回:
    - yield_prediction: 预测产量(吨/公顷)
    """
    base_yield = 5.0
    
    temp_factor = 1.0 - Abs(temp - 25) / 25
    rainfall_factor = 1.0 - Abs(rainfall - 600) / 600
    fertilizer_factor = 1.0 + fertilizer / 200
    sunlight_factor = 0.8 + sunlight / 12 * 0.4
    soil_factor = 1.0 + soil_quality
    
    yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor
    return max(0, yield_prediction)  # 产量不能为负


@mcp.tool()
def multidimensional_evaluation(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float
) -> float:
    """
    多维指标综合评分函数
    用于对多维指标进行加权计算，输出统一的数值评估结果
    
    参数:
    - a, b, c, d, e: 五个独立的实数指标
    
    返回:
    - 结果: 实数值评估结果
    """
    if e == 0:
        raise ValueError("变量 e 不能为零")
    
    result = 2*a + 3*b**2 - 4*c + d/e + 5
    return result


@mcp.tool()
def sdoF_system_response(
    m: float,
    c: float,
    k: float,
    u_g: float
) -> float:
    """
    单自由度系统动力响应模型
    用于模拟结构在地震作用下的动力响应
    
    参数:
    - m: 系统的质量
    - c: 阻尼系数
    - k: 刚度
    - u_g: 地面的地震加速度输入
    
    返回:
    - u: 结构相对于地面的位移响应
    """
    # 简化的计算模型，实际应使用微分方程求解
    # 这里仅作演示用途
    u = -m * u_g / (k + 1e-10)  # 添加小量防止除零
    return u


@mcp.tool()
def chemical_reaction_efficiency(
    x1: float,
    x2: float,
    x3: float
) -> float:
    """
    化工反应效率预测模型
    基于原料浓度、温度条件和压力条件预测反应效率
    
    参数:
    - x1: 原料A的浓度(%)
    - x2: 温度参数(℃)
    - x3: 压力参数(MPa)
    
    返回:
    - y: 预测的反应效率或产出值
    """
    beta_1, beta_2, beta_3, beta_0 = 2.5, 1.8, -0.7, 3.2
    y = beta_1 * x1 + beta_2 * x2 + beta_3 * x3 + beta_0
    return y


@mcp.tool()
def fuel_consumption_prediction(
    speed: float,
    weight: float
) -> float:
    """
    汽车油耗预测模型
    基于行驶速度和车辆载重预测百公里油耗
    
    参数:
    - speed: 行驶速度(km/h)
    - weight: 汽车载重(kg)
    
    返回:
    - fuel_consumption: 百公里油耗(L/100km)
    """
    fuel_consumption = 0.05 * speed + 0.002 * weight + 3
    return fuel_consumption


@mcp.tool()
def crop_growth_rate(
    temp: float,
    rainfall: float,
    sunlight: float,
    soil_nutrient: float
) -> float:
    """
    作物生长速率模型
    基于环境因子预测作物生长速率
    
    参数:
    - temp: 温度(℃)
    - rainfall: 降水量(mm/天)
    - sunlight: 每日光照时长(小时)
    - soil_nutrient: 土壤养分含量(mg/kg)
    
    返回:
    - dW_dt: 作物生物量随时间的变化率(kg/天)
    """
    k, a, b, c = 1.0, 0.1, 0.01, 0.02  # 经验系数
    dW_dt = k * (1 - exp(-a * temp)) * (1 - exp(-b * rainfall)) * sunlight * (1 - exp(-c * soil_nutrient))
    return dW_dt


@mcp.tool()
def traffic_speed_dynamics(
    v_free: float,
    k: float,
    q_max: float,
    alpha: float = 0.1,
    beta: float = 0.05
) -> float:
    """
    交通流速度动态模型
    描述交通流速度随时间的变化率
    
    参数:
    - v_free: 自由流速度(km/h)
    - k: 车流密度(辆/km)
    - q_max: 道路最大通行能力(辆/h)
    - alpha: 速度恢复系数(1/s)
    - beta: 密度阻尼系数
    
    返回:
    - dv_dt: 速度变化率(km/h/s)
    """
    v = 50  # 假设当前平均速度为50km/h
    dv_dt = alpha * (v_free - v) - beta * k / q_max * v
    return dv_dt


@mcp.tool()
def chemical_process_efficiency(
    T: float,
    P: float,
    C: float,
    t: float,
    k: float
) -> float:
    """
    化工过程效率模型
    基于反应温度、压力、反应物浓度、反应时间和催化剂活性预测效率
    
    参数:
    - T: 反应温度
    - P: 压力
    - C: 反应物浓度
    - t: 反应时间
    - k: 催化剂活性
    
    返回:
    - Efficiency: 效率值(0-1)
    """
    a1, a2, a3, a4 = 0.01, 0.001, 0.1, 0.05  # 经验参数
    exponent = a1 * T + a2 * P * C + a3 * log(t + 1) + a4 * k**2
    Efficiency = 1 / (1 + exp(-exponent))
    return Efficiency


@mcp.tool()
def cultural_influence_growth(
    t: float,
    audience_growth_rate: float,
    engagement_factor: float,
    k: float = 0.8
) -> float:
    """
    文化传播影响力增长模型
    描述影响力随时间的动态变化
    
    参数:
    - t: 传播持续时间(天)
    - audience_growth_rate: 受众增长率(人/天)
    - engagement_factor: 互动系数[0,1]
    - k: 标准化系数
    
    返回:
    - dI_dt: 影响力变化率(影响力/天)
    """
    dI_dt = k * log(t + 1) * audience_growth_rate * engagement_factor
    return dI_dt


@mcp.tool()
def crop_growth_dynamics(
    W: float,
    N: float,
    T: float
) -> float:
    """
    作物生长动态模型
    基于土壤含水量、氮含量和温度预测作物生长速率
    
    参数:
    - W: 土壤含水量
    - N: 土壤氮含量
    - T: 环境温度(℃)
    
    返回:
    - dG_dt: 单位时间作物生物量的变化速率(生物量/天)
    """
    # 使用Python内置abs函数替代sympy的Abs函数
    dG_dt = 0.05 * W * (1 - exp(-0.1 * N)) * (1 - abs(T - 25) / 20)
    print("当前温度为：", T)
    print("当前含水量为：", W)
    print("当前氮含量为：", N)
    print("当前作物生长速率为：", dG_dt)
    # 转换为标准Python float类型
    return float(dG_dt)


@mcp.tool()
def integral_calculator(
    expression: str,
    variable: str,
    lower_limit: float = None,
    upper_limit: float = None
) -> str:
    """
    积分计算器
    支持定积分和不定积分计算
    
    参数:
    - expression: 数学表达式字符串（例如："x**2 + 2*x + 1"）
    - variable: 积分变量（例如："x"）
    - lower_limit: 积分下限（可选，为定积分）
    - upper_limit: 积分上限（可选，为定积分）
    
    返回:
    - result: 积分结果的字符串表示
    """
    try:
        # 定义符号变量
        x = symbols(variable)
        
        # 解析表达式
        expr = sp.sympify(expression)
        
        # 计算积分
        if lower_limit is not None and upper_limit is not None:
            # 定积分
            result = integrate(expr, (x, lower_limit, upper_limit))
            return f"定积分 ∫({expression})d{variable} 从 {lower_limit} 到 {upper_limit} = {result}"
        else:
            # 不定积分
            result = integrate(expr, x)
            return f"不定积分 ∫({expression})d{variable} = {result} + C"
    except Exception as e:
        return f"积分计算出错: {str(e)}"


if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run(transport="http", port=8888)
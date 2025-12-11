import numpy as np
from fastmcp import FastMCP

# 创建MCP服务器实例
mcp = FastMCP(
    name="Numpy-Formula-Calculator-MCP",
    instructions="This server contains tools for calculating mathematical formulas using numpy."
)

@mcp.tool
def calculate_cow_milk_production(
    feed_quality: float,
    temperature: float,
    health_score: float,
    drinking_time: float
) -> float:
    """
    根据奶牛饲养条件计算日均产奶量
    
    公式: 产奶量 = 20 + 1.5 × 饲料质量 - 0.3 × (气温 - 20) + 0.05 × 健康评分 + 0.4 × 饮水时间 + ε
    其中 ε 是服从 N(0,1) 的随机扰动项，这里取期望值0
    """
    milk_yield = 20 + 1.5 * feed_quality - 0.3 * (temperature - 20) + 0.05 * health_score + 0.4 * drinking_time
    return float(milk_yield)

@mcp.tool
def calculate_crop_yield_by_fertilizer(fertilizer_amount: float) -> float:
    """
    根据施肥量计算作物产量
    
    公式: 产量 = 500 + 40 × x - 0.5 × x²
    其中 x 是施肥量（kg/亩）
    """
    yield_amount = 500 + 40 * fertilizer_amount - 0.5 * fertilizer_amount**2
    return float(yield_amount)

@mcp.tool
def calculate_ev_battery_level(prev_battery_level: float, external_input: float) -> float:
    """
    计算电动汽车当前电池电量
    
    公式: B_t = 0.95 × B_{t-1} - 1.2 + u_t
    其中 B_t 是当前电量，B_{t-1} 是前一时刻电量，u_t 是外部输入
    """
    current_battery = 0.95 * prev_battery_level - 1.2 + external_input
    return float(current_battery)

@mcp.tool
def calculate_crop_growth_rate(
    temperature: float,
    irrigation_amount: float,
    coefficient_a: float = 0.5,
    coefficient_b: float = 0.3,
    coefficient_c: float = 0.02
) -> float:
    """
    计算作物生长变化量
    
    公式: ΔG_t = a×T_t + b×W_t - c×(T_t×W_t)
    其中 T_t 是平均温度，W_t 是灌溉水量
    """
    growth_change = coefficient_a * temperature + coefficient_b * irrigation_amount - coefficient_c * (temperature * irrigation_amount)
    return float(growth_change)

@mcp.tool
def calculate_concrete_deformation(
    stress: float,
    elastic_modulus: float,
    time_constant: float,
    time_days: float = 365
) -> float:
    """
    计算混凝土柱长期变形（轴向应变）
    
    公式: ε(t) = (σ/E) × (t - τ×ln(1 + t/τ))
    其中 σ 是轴向应力，E 是弹性模量，τ 是时间常数，t 是时间
    """
    strain = (stress / elastic_modulus) * (time_days - time_constant * np.log(1 + time_days / time_constant))
    return float(strain)

@mcp.tool
def calculate_vehicle_acceleration(
    max_force: float,
    throttle: float,
    resistance: float,
    mass: float
) -> float:
    """
    计算车辆加速度
    
    公式: a = (F - R) / m
    其中 F = F_max × throttle 是驱动力，R 是阻力，m 是质量
    """
    force = max_force * throttle
    acceleration = (force - resistance) / mass
    return float(acceleration)

@mcp.tool
def calculate_traffic_flow(
    density: float,
    speed: float,
    time: float
) -> float:
    """
    计算交通流量
    
    公式: Traffic Flow = density × speed × time
    """
    traffic_flow = density * speed * time
    return float(traffic_flow)

@mcp.tool
def calculate_nonlinear_function(x: float, y: float) -> float:
    """
    计算非线性多变量函数值
    
    公式: f(x,y) = x² + 2×sin(y) + e^(0.5×x×y)
    """
    result = x**2 + 2 * np.sin(y) + np.exp(0.5 * x * y)
    return float(result)

@mcp.tool
def calculate_concrete_column_deformation(
    stress: float,
    elastic_modulus: float,
    time_constant: float,
    time_days: float = 180
) -> float:
    """
    计算混凝土柱长期变形（另一种模型）
    
    公式: ε(t) = (σ/E) × (t - τ×ln(1 + t/τ))
    """
    strain = (stress / elastic_modulus) * (time_days - time_constant * np.log(1 + time_days / time_constant))
    return float(strain)

@mcp.tool
def calculate_agricultural_yield(
    base_yield: float,
    temperature: float,
    rainfall: float,
    fertilizer: float,
    temp_threshold: float = 10,
    temp_coeff: float = 0.5,
    rain_coeff: float = 0.01,
    fert_coeff: float = 0.2
) -> float:
    """
    计算农业产量预测
    
    公式: Y = Y₀ + β_T × max(T-T_base, 0) + β_R × R + β_F × F + ε×(...)
    这里简化为: Y = Y₀ + β_T × max(T-T_base, 0) + β_R × R + β_F × F
    """
    temp_effect = temp_coeff * max(temperature - temp_threshold, 0)
    yield_prediction = base_yield + temp_effect + rain_coeff * rainfall + fert_coeff * fertilizer
    return float(yield_prediction)
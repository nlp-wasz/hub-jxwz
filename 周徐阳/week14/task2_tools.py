import math
from typing import Optional

import numpy as np
import sympy as sp

def get_bmr_rate(weight: float, height: float):
    """
    在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理
功能所需能量的重要指标。它广泛应用于能量需求评估、体重管理、临床营养支持等多个场
景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，
构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所
需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算。
    :param weight: 个体体重，单位为千克（kg）；
    :param height: 个体身高，单位为厘米（cm）。
    :return: 基础代谢率（BMR）
    """
    return round(10 * weight + 6.25 * height - 100, 2)


def simulate_retail_revenue(
    input_kg: float,
    base_price: float,
    fluctuation_min: float,
    fluctuation_max: float,
    seed: Optional[int] = None,
) -> float:
    """
    零售收入模型：Revenue = Input_kg * (Base_price + Fluctuation)，Fluctuation 服从均匀分布。
    通过设置随机种子可以复现带有价格波动的模拟结果。
    """
    rng = np.random.default_rng(seed)
    fluctuation = rng.uniform(fluctuation_min, fluctuation_max)
    revenue = input_kg * (base_price + fluctuation)
    return float(revenue)


def estimate_monthly_rent(
    area: float,
    location_score: float,
    age: float,
    bedrooms: int,
    distance_to_subway: float,
) -> float:
    """
    房产月租金估算模型：
    Monthly Rent = 50 * Area * (LocationScore/10) * (1 - Age/30) * (1 + 0.1*Bedrooms) * (1000/(1000 + DistanceToSubway))
    线性与非线性调节项结合，评估面积、区位、房龄、卧室数量与地铁距离的影响。
    """
    age_factor = max(0.0, 1 - age / 30)
    location_factor = max(location_score / 10, 0.0)
    distance_factor = 1000 / (1000 + max(distance_to_subway, 0.0))
    rent = (
        50.0
        * area
        * location_factor
        * age_factor
        * (1 + 0.1 * bedrooms)
        * distance_factor
    )
    return float(rent)


def nonlinear_interaction(x: float, y: float) -> float:
    """
    非线性交互模型：fun(x, y) = 2.5*sin(x) + 1.8*cos(y) + 0.3*x*y。
    用于模拟两个输入变量的周期性与乘积交互效应。
    """
    return float(2.5 * math.sin(x) + 1.8 * math.cos(y) + 0.3 * x * y)


def dissolved_oxygen_rate(
    concentration: float,
    second_derivative: float,
    diffusion_coeff: float,
    exchange_rate: float,
    saturation_conc: float,
    bio_load: float,
    temp_c: float,
    temp_sensitivity: float = 0.02,
) -> float:
    """
    溶解氧一维扩散-反应框架：
    dC/dt = D*d2C/dx2 - k*(C - C_sat) + R(bio_load, temp)。
    second_derivative 为 d2C/dx2 的近似值，R 简化为 -bio_load * (1 + temp_sensitivity*(temp-20)).
    """
    respiration = -bio_load * (1 + temp_sensitivity * (temp_c - 20.0))
    dcdt = diffusion_coeff * second_derivative - exchange_rate * (
        concentration - saturation_conc
    ) + respiration
    return float(dcdt)


def inventory_step(
    current_stock: float, decay_rate: float, replenishment: float, dt: float = 1.0
) -> float:
    """
    库存动态：dS/dt = -k*S + u(t) 的显式欧拉更新。
    dt 为时间步长，返回下一时刻库存量并避免出现负库存。
    """
    delta = (-decay_rate * current_stock + replenishment) * dt
    next_stock = max(current_stock + delta, 0.0)
    return float(next_stock)


def predict_milk_yield(
    feed_quality: float,
    health_status: float,
    avg_temp: float,
    milk_freq: float,
    lactation_week: float,
) -> float:
    """
    奶牛日均产奶量预测：
    yield = 25*(feed_quality/100)*(health_status/100)*(1 - 0.05*|avg_temp-20|)*(milk_freq/2)*(1 - exp(-0.1*lactation_week)).
    温度项与泌乳周期项会进行下限截断以避免出现负值。
    """
    temp_factor = max(0.0, 1 - 0.05 * abs(avg_temp - 20))
    cycle_factor = max(0.0, 1 - math.exp(-0.1 * lactation_week))
    yield_kg = (
        25.0
        * (feed_quality / 100.0)
        * (health_status / 100.0)
        * temp_factor
        * (milk_freq / 2.0)
        * cycle_factor
    )
    return float(max(yield_kg, 0.0))


def complex_system_response(
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    x5: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
) -> float:
    """
    复杂系统输出：
    y = ∫_0^{x1} a*exp(-t/b)/(c+t) dt + d*x2^2 + e*sin(x3) + log(x4+1) + sqrt(x5)。
    使用 sympy 进行积分，针对对数与平方根项做非负截断以避免无效输入。
    """
    t = sp.symbols("t", real=True)
    integral_expr = sp.integrate(a * sp.exp(-t / b) / (c + t), (t, 0, x1))
    integral_val = float(sp.N(integral_expr))
    log_term = math.log(max(x4 + 1.0, 1e-12))
    sqrt_term = math.sqrt(max(x5, 0.0))
    result = integral_val + d * (x2**2) + e * math.sin(x3) + log_term + sqrt_term
    return float(result)


def next_body_weight(
    current_weight: float, calorie_intake: float, calorie_expenditure: float, k: float
) -> float:
    """
    体重差分模型：W_{t+1} = W_t + (C_t - E_t)/k。
    k 表示热量与体重的转换系数，需为非零正数。
    """
    if k == 0:
        raise ValueError("k must be non-zero to compute weight change.")
    next_w = current_weight + (calorie_intake - calorie_expenditure) / k
    return float(next_w)


def discounted_derivative_price(s: float, k: float, r: float, t: float) -> float:
    """
    简化的金融衍生品定价：price = exp(-r*T) * max(S - K, 0)。
    体现 Black-Scholes 框架中的贴现与非负性约束。
    """
    discounted_intrinsic = math.exp(-r * t) * max(s - k, 0.0)
    return float(discounted_intrinsic)


def average_daily_gain(feed_intake: float, protein_content: float, animal_weight: float) -> float:
    """
    日增重模型：ADG = (feed_intake * protein_content) / (animal_weight * 10)。
    当体重为零或负值时抛出异常以避免除零错误。
    """
    if animal_weight <= 0:
        raise ValueError("animal_weight must be positive.")
    adg = (feed_intake * protein_content) / (animal_weight * 10.0)
    return float(adg)

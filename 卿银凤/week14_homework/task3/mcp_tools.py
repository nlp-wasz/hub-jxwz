import numpy as np
import math
from sympy import symbols, diff, exp, Function
from sympy.abc import t, z


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


def get_property_value(area: float, floor: float, age: float):
    """
    在房地产市场中，房产的市场价值受多种因素影响，包括面积、楼层和房龄等。
    该模型构建一个经验型评估模型，用于估算房产的市场价值，有助于辅助定价、投资决策和市场分析。
    :param area: 房产面积，单位：平方米
    :param floor: 楼层系数，无量纲
    :param age: 房龄，单位：年
    :return: 房产市场价值
    """
    value = 10000 * area * (1 + 0.02 * floor) * (1 - 0.015 * age)
    return round(value, 2)


def lake_pollutant_change_rate(C: float, Qin: float, Cin: float, V: float, k: float):
    """
    在环境工程与生态建模中，理解与预测水体中污染物浓度的变化趋势是评估水质状况的关键环节。
    该模型通过建立常微分方程来刻画污染物浓度随时间的变化率，考虑外部输入与内部降解过程。
    :param C: 湖泊中当前污染物浓度（单位：mg/L）
    :param Qin: 进水流量（单位：m³/天）
    :param Cin: 进水污染物浓度（单位：mg/L）
    :param V: 湖泊总体积（单位：m³）
    :param k: 污染物自然降解速率（单位：1/天）
    :return: 污染物浓度变化率 dC/dt（单位：mg/L/天）
    """
    dC_dt = (Qin * (Cin - C)) / V - k * C
    return round(dC_dt, 4)


def linear_model_output(x1: float, x2: float, x3: float, x4: float, x5: float):
    """
    线性模型是一种基础但非常重要的建模工具，广泛应用于预测分析、系统建模和数据趋势估计等领域。
    该模型通过构建输入变量与输出结果之间的线性关系，可以有效地对复杂系统进行简化描述。
    :param x1: 输入变量1
    :param x2: 输入变量2
    :param x3: 输入变量3
    :param x4: 输入变量4
    :param x5: 输入变量5
    :return: 模型输出值
    """
    w1, w2, w3, w4, w5 = 1.2, -0.5, 3.0, 2.1, -1.0
    b = 4.5
    y = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + b
    return round(y, 2)


def combustion_efficiency(HV: float):
    """
    在能源化工领域，燃烧过程的效率是评估燃料性能和系统优化的关键指标之一。
    该模型基于经验数据，建立了燃料热值与燃烧效率之间的定量线性关系。
    :param HV: 燃料热值，单位：MJ/kg，范围：40 ≤ HV ≤ 50
    :return: 燃烧效率 η，无量纲（范围：0-1）
    """
    if HV <= 40 or HV >= 50:
        raise ValueError("热值必须在40到50 MJ/kg之间")

    eta = 0.85 + 0.005 * (HV - 40)
    return round(eta, 4)


def fish_population_dynamics(P_t: float, r: float, K: float, S: float):
    """
    在水产养殖管理中，科学预测和控制鱼类种群数量对于实现可持续发展具有重要意义。
    该模型采用一阶差分方程描述鱼类种群在离散时间步长下的动态演化，考虑自然增长与人为投放。
    :param P_t: 第t个时间步的鱼类种群数量（单位：千尾）
    :param r: 种群内部增长率，无量纲
    :param K: 环境承载力（单位：千尾）
    :param S: 每时间步新增投放的鱼苗数量（单位：千尾）
    :return: 第t+1个时间步的鱼类种群数量（单位：千尾）
    """
    P_t1 = P_t + 1
    P_t1 = P_t + r * P_t * (1 - P_t / K) + S
    return round(P_t1, 2)


def soil_moisture_content(theta0: float, k: float, depth: float):
    """
    在农业和环境科学研究中，理解土壤中水分的分布与迁移机制对于作物生长和灌溉管理具有重要意义。
    该模型描述在稳态条件下，土壤含水量随深度变化的指数衰减关系。
    :param theta0: 地表初始含水量，无量纲
    :param k: 衰减系数，反映水分随深度衰减的速率（单位：1/m）
    :param depth: 土壤深度（单位：m）
    :return: 深度z处的土壤含水量，无量纲
    """
    theta = theta0 * np.exp(-k * depth)
    return round(theta, 4)


def texture_hardness_prediction(temp: float, time: float, ph: float, sugar: float):
    """
    在食品加工与制造过程中，成品的质地硬度是一个关键的质量控制指标。
    该模型基于关键工艺参数（温度、时间、pH值、糖分含量）预测食品质地硬度。
    :param temp: 加工温度（单位：℃）
    :param time: 加工时间（单位：分钟）
    :param ph: 原料pH值（无量纲）
    :param sugar: 糖分含量（单位：g/100g）
    :return: 质地硬度预测值
    """
    hardness = 0.5 * temp + 1.2 * time - 3.0 * ph + 0.8 * sugar
    return round(hardness, 2)


def environmental_quality_index(pollution_level: float, population_density: float, green_coverage: float):
    """
    环境质量指数是一种综合评估特定区域环境健康状况的指标。
    该模型通过定量分析污染水平、绿化覆盖率以及人口密度三者之间的关系，反映环境承载能力和宜居水平。
    :param pollution_level: 污染水平，范围：0-100
    :param population_density: 人口密度（单位：人/平方公里）
    :param green_coverage: 绿化覆盖率，范围：0-1
    :return: 环境质量指数（范围：0-100），数值越高表示环境质量越好
    """
    EQI = (100 - pollution_level) / (1 + population_density * (1 - green_coverage))
    return round(EQI, 2)


def fish_daily_growth_random(feed_intake: float, water_temp: float, stocking_density: float, oxygen_level: float):
    """
    在水产养殖过程中，鱼类的生长受到多种环境与管理因素的综合影响。
    该模型模拟鱼类的日增重，综合考虑水温、饲料摄入量、溶解氧浓度、养殖密度等关键变量，并引入随机扰动项。
    :param feed_intake: 饲料摄入量（单位：g/尾/天）
    :param water_temp: 水温（单位：℃）
    :param stocking_density: 养殖密度（单位：尾/m³）
    :param oxygen_level: 溶解氧浓度（单位：mg/L）
    :return: 日增重预测值（单位：g/天），包含随机扰动
    """
    # 基础日增重计算
    base_ADG = 0.1 * feed_intake + 0.05 * water_temp - 0.02 * stocking_density + 0.03 * oxygen_level

    # 添加随机扰动项，服从均值为0、标准差为0.05的正态分布
    epsilon = np.random.normal(0, 0.05)

    ADG = base_ADG + epsilon
    return round(ADG, 3)
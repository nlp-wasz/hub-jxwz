from fastmcp import FastMCP
import numpy as np
import math

mcp=FastMCP(
    name="tools-MCP-Server",
    instructions="""This server contains some calculation formulas"""
)

@mcp.tool()
def get_bmr_rate(weight: float, height: float):
    """
    计算个体在静息状态下维持基本生理功能所需能量的重要指标，基于体重和身高数据快速估算其每日基础代谢所需热量
    :param weight: 个体体重，单位为千克（kg）；
    :param height: 个体身高，单位为厘米（cm）。
    :return: 基础代谢率（BMR
    """
    return round(10 * weight + 6.25 * height - 100, 2)

@mcp.tool()
def calculate_complex_system(x: float, y: float):
    """
    描述多变量输入的复杂系统，包含随时间或空间累积的动态效应
    :param x: 第一个输入变量
    :param y: 第二个输入变量
    :return: 系统输出值
    """
    # 计算系统输出
    result = 2.5 * math.sin(x) + 1.8 * math.cos(y) + 0.3 * x * y

    # 返回结果，保留两位小数
    return round(result, 2)

@mcp.tool()
def calculate_deterministic_model(x1: float, x2: float, x3: float, x4: float):
    """
    基于已知数学关系，将多个输入变量通过预定义公式转换为输出结果
    :param x1: 第一个输入变量
    :param x2: 第二个输入变量
    :param x3: 第三个输入变量
    :param x4: 第四个输入变量
    :return: 系统输出值
    """
    # 计算系统输出
    y = 2 * x1 + 3 * x2 - 1.5 * x3 + 0.8 * x4 + math.exp(0.1 * x2)
    # 返回结果，保留两位小数
    return round(y, 2)


@mcp.tool()
def calculate_monthly_rent(area: float, location_score: float, age: float, bedrooms: float, distance_to_subway: float):
    """
    预测房产的潜在月租金收益，考虑面积、地段评分、房龄、卧室数量和交通便利性等因素

     :param area: 房产面积（单位：平方米）
    :param location_score: 地段评分（1-10分）
    :param age: 房龄（单位：年）
    :param bedrooms: 卧室数量
    :param distance_to_subway: 距离地铁的距离（单位：米）
    :return: 预测的月租金（单位：元）
    """
    # 计算月租金
    monthly_rent = 50 * area * (location_score / 10) * (1 - age / 30) * (1 + 0.1 * bedrooms) * (
                1000 / (1000 + distance_to_subway))

    # 返回结果，保留两位小数
    return round(monthly_rent, 2)

@mcp.tool()
def calculate_do_concentration_change(C: float, D: float, k: float, C_sat: float, bio_load: float, temp: float,
                                      x: float, t: float):
    """
    基于扩散-反应机制，预测水产养殖系统中溶解氧浓度的动态变化

    :param C: 当前溶解氧浓度
    :param D: 扩散系数
    :param k: 气体交换速率系数
    :param C_sat: 温度相关的溶解氧饱和浓度
    :param bio_load: 生物负载
    :param temp: 水温
    :param x: 空间位置
    :param t: 时间
    :return: 溶解氧浓度变化率
    """
    # 简化的生物耗氧速率函数
    R = 0.01 * bio_load * (1 + 0.05 * (temp - 20))
    # 计算溶解氧浓度变化率
    dC_dt = D * 0 - k * (C - C_sat) + R  # 简化空间二阶导数为0
    # 返回结果，保留四位小数
    return round(dC_dt, 4)

@mcp.tool()
def calculate_milk_production(feed_quality: float, health_status: float, avg_temp: float, milk_freq: float,
                              lactation_week: float):
    """
    预测奶牛的日均产奶量，考虑饲料质量、健康状况、泌乳周期、环境温度和挤奶频率等因素

     :param feed_quality: 饲料质量（0-100分）
    :param health_status: 健康状态（0-100分）
    :param avg_temp: 平均环境温度（℃）
    :param milk_freq: 每日挤奶频率
    :param lactation_week: 泌乳周数
    :return: 预测的日均产奶量（单位：kg）
    """
    # 计算产奶量
    milk_production = 25 * (feed_quality / 100) * (health_status / 100) * (1 - 0.05 * abs(avg_temp - 20)) * (
                milk_freq / 2) * (1 - math.exp(-0.1 * lactation_week))
    # 返回结果，保留两位小数
    return round(milk_production, 2)


@mcp.tool()
def predict_crop_yield(temp: float, rainfall: float, fertilizer: float, sunlight: float, soil_quality: float,
                       base_yield: float = 5.0):
    """
    预测农作物产量，考虑温度、降水量、施肥量、光照时长和土壤质量等因素
     :param temp: 平均生长温度（℃）
    :param rainfall: 生长期间降水量（mm）
    :param fertilizer: 施肥量（kg/公顷）
    :param sunlight: 每日平均光照时长（小时）
    :param soil_quality: 土壤质量指数（0-1之间）
    :param base_yield: 基础产量水平（默认5.0吨/公顷）
    :return: 预测的作物产量（吨/公顷）
    """
    # 计算各影响因子
    temp_factor = max(0, 1.0 - abs(temp - 25) / 25)
    rainfall_factor = max(0, 1.0 - abs(rainfall - 600) / 600)
    fertilizer_factor = 1.0 + fertilizer / 200
    sunlight_factor = 0.8 + (sunlight / 12) * 0.4
    soil_factor = 1.0 + soil_quality

    # 计算预测产量
    yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor

    # 确保产量不为负
    yield_prediction = max(0, yield_prediction)

    # 返回结果，保留两位小数
    return round(yield_prediction, 2)

@mcp.tool()
def predict_daily_sales(price_per_kg: float, supply_kg: float, demand_factor: float,
                        discount_rate: float, weather_factor: float):
    """
    预测农产品的日销售总额，考虑价格、供应量、市场需求、促销活动以及天气等因素的影响
    :param price_per_kg: 每公斤农产品的销售价格（单位：元）
    :param supply_kg: 当日可供应的农产品总量（单位：公斤）
    :param demand_factor: 需求系数，反映市场对当前农产品的需求强度，取值范围为 0 到 1
    :param discount_rate: 折扣率，表示促销活动对价格的下调比例，取值范围为 0 到 1
    :param weather_factor: 天气影响因子，用于量化天气状况对销售的影响，取值范围通常在 0.8 至 1.2 之间
    :return: 预测的日销售总额（单位：元）
    """
    # 参数验证
    if not (0 <= demand_factor <= 1):
        raise ValueError("需求系数(demand_factor)必须在0到1之间")
    if not (0 <= discount_rate <= 1):
        raise ValueError("折扣率(discount_rate)必须在0到1之间")
    if not (0.8 <= weather_factor <= 1.2):
        raise ValueError("天气影响因子(weather_factor)必须在0.8到1.2之间")

    # 计算日销售额
    daily_sales = price_per_kg * supply_kg * demand_factor * (1 - discount_rate) * weather_factor

    # 返回结果，保留两位小数
    return round(daily_sales, 2)


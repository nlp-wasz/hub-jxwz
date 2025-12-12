import numpy as np
import math
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Dairy Cow Milk Production Model")

@mcp.tool()
def calculate_daily_milk_yield(
    feed_quality: float = 85.0,        # 饲料质量 (百分比)
    health_status: float = 90.0,       # 健康状态 (百分比)
    avg_temp: float = 20.0,            # 环境温度 (°C)
    milk_freq: float = 2.0,            # 挤奶频率 (次/天)
    lactation_week: float = 10.0       # 泌乳周期 (周)
) -> dict:
    """
    奶牛日均产奶量预测模型
    
    建模背景:
    在现代畜牧业中，奶牛的日均产奶量是衡量养殖场生产效率和动物健康管理质量的重要指标。
    为了实现科学饲养与精细化管理，建立一个能够反映关键影响因素的产奶量预测模型具有重要意义。
    该模型综合考虑了饲料质量、健康状况、泌乳周期、环境温度以及挤奶频率等五个关键变量，
    旨在模拟奶牛在不同饲养条件下的产奶潜力。
    
    模型设计基于确定性建模思路，即在已知输入变量的情况下，通过设定的数学关系
    明确计算出日均产奶量。该方法具有可解释性强、计算效率高、便于集成至生产管理系统等优点。
    通过此模型，养殖管理者可以评估当前饲养条件对产奶性能的影响，并据此优化饲养策略，提高生产效益。
    
    建模公式:
    \[ 产奶量 = 25 \times \left( \frac{feed\_quality}{100} \right) \times \left( \frac{health\_status}{100} \right) \times (1 - 0.05 \times |avg\_temp - 20|) \times \left( \frac{milk\_freq}{2} \right) \times (1 - e^{-0.1 \times lactation\_week}) \]
    
    该公式以25kg为基础产奶量，结合多个影响因子对产奶量进行逐层调整。其中：
    - **饲料质量因子** 反映营养供给水平对产奶能力的支持程度；
    - **健康状态因子** 体现奶牛生理状况对产奶表现的影响；
    - **温度响应因子** 用于模拟环境温度偏离最适范围时的产奶抑制效应；
    - **挤奶频率因子** 表示挤奶次数对产奶潜力的正向激励；
    - **泌乳周期因子** 刻画奶牛在整个泌乳期中产奶量的变化趋势。
    
    该模型在结构上体现了各因素之间的交互作用，能够有效反映饲养管理、环境条件与生理阶段
    对奶牛产奶性能的综合影响，适合作为牧场管理决策支持系统的组成部分。
    
    参数说明:
    - feed_quality: 饲料质量，表示饲料的营养水平，范围：0-100 (百分比)
    - health_status: 健康状态，表示奶牛的整体健康状况，范围：0-100 (百分比)
    - avg_temp: 环境温度，表示奶牛所处环境的平均温度，单位：°C
    - milk_freq: 挤奶频率，表示每天挤奶的次数，单位：次/天
    - lactation_week: 泌乳周期，表示奶牛产犊后的周数，单位：周
    
    返回:
    - dict: 包含产奶量、各因子贡献值、模型参数和建议的字典
    
    使用示例:
    >>> result = calculate_daily_milk_yield(
    ...     feed_quality=85.0,
    ...     health_status=90.0,
    ...     avg_temp=20.0,
    ...     milk_freq=2.0,
    ...     lactation_week=10.0
    ... )
    >>> print(f"日均产奶量: {result['milk_yield']:.2f} kg/天")
    """
    
    # 1. 参数验证
    if feed_quality < 0 or feed_quality > 100:
        raise ValueError("饲料质量应在0-100之间")
    if health_status < 0 or health_status > 100:
        raise ValueError("健康状态应在0-100之间")
    if milk_freq <= 0:
        raise ValueError("挤奶频率应大于0")
    if lactation_week < 0:
        raise ValueError("泌乳周期不能为负数")
    
    # 2. 计算各因子
    # 基础产奶量
    base_yield = 25.0
    
    # 饲料质量因子
    feed_factor = feed_quality / 100.0
    
    # 健康状态因子
    health_factor = health_status / 100.0
    
    # 温度响应因子
    temp_deviation = abs(avg_temp - 20.0)
    # 温度偏离导致的产奶抑制，最大抑制为0.05*100=5%，即温度偏离20°C越多，抑制越强
    # 公式中的0.05是每度偏离的抑制系数
    temp_factor = 1.0 - 0.05 * temp_deviation
    # 确保温度因子不会变为负数
    temp_factor = max(temp_factor, 0.0)
    
    # 挤奶频率因子
    milk_freq_factor = milk_freq / 2.0
    
    # 泌乳周期因子
    lactation_factor = 1.0 - math.exp(-0.1 * lactation_week)
    
    # 3. 计算总产奶量
    milk_yield = base_yield * feed_factor * health_factor * temp_factor * milk_freq_factor * lactation_factor
    
    # 4. 计算各因子对产奶量的贡献度（相对于最优情况）
    # 最优情况下的因子值
    optimal_feed_factor = 1.0
    optimal_health_factor = 1.0
    optimal_temp_factor = 1.0  # 温度为20°C时
    optimal_milk_freq_factor = 1.0  # 每天挤奶2次时
    # 泌乳周期因子随周数增长，我们取第30周作为泌乳高峰参考
    optimal_lactation_factor = 1.0 - math.exp(-0.1 * 30)
    
    # 各因子的相对贡献（当前因子值 / 最优因子值）
    feed_contribution = feed_factor / optimal_feed_factor * 100
    health_contribution = health_factor / optimal_health_factor * 100
    temp_contribution = temp_factor / optimal_temp_factor * 100
    milk_freq_contribution = milk_freq_factor / 1.0 * 100  # 每天挤奶2次为基准
    lactation_contribution = lactation_factor / optimal_lactation_factor * 100
    
    # 5. 识别主要限制因素
    contributions = {
        "饲料质量": feed_contribution,
        "健康状况": health_contribution,
        "温度条件": temp_contribution,
        "挤奶频率": milk_freq_contribution,
        "泌乳阶段": lactation_contribution
    }
    
    min_contribution_factor = min(contributions, key=contributions.get)
    min_contribution_value = contributions[min_contribution_factor]
    
    # 6. 生成管理建议
    suggestions = []
    
    if feed_contribution < 80:
        suggestions.append(f"饲料质量较低 ({feed_quality:.1f}%)，建议提高饲料营养水平")
    
    if health_contribution < 80:
        suggestions.append(f"健康状况需改善 ({health_status:.1f}%)，建议加强疾病预防和健康监测")
    
    if temp_contribution < 80:
        suggestions.append(f"温度条件不理想 ({avg_temp:.1f}°C)，建议调整环境温度至20°C左右")
    
    if milk_freq_contribution < 80:
        suggestions.append(f"挤奶频率较低 ({milk_freq:.1f}次/天)，建议增加挤奶次数")
    
    if lactation_contribution < 50:
        suggestions.append(f"处于泌乳初期 (第{lactation_week:.1f}周)，产奶量尚未达到高峰")
    elif lactation_contribution > 90:
        suggestions.append(f"处于泌乳中后期 (第{lactation_week:.1f}周)，产奶量接近或达到高峰")
    
    if not suggestions:
        suggestions.append("当前饲养条件良好，继续保持")
    
    # 7. 计算潜在提升空间（相对于最优条件）
    optimal_yield = base_yield * 1.0 * 1.0 * 1.0 * 1.0 * optimal_lactation_factor
    improvement_potential = optimal_yield - milk_yield
    improvement_percentage = (improvement_potential / milk_yield) * 100 if milk_yield > 0 else 0
    
    return {
        "milk_yield": float(milk_yield),
        "base_yield": float(base_yield),
        "factors": {
            "feed_factor": float(feed_factor),
            "health_factor": float(health_factor),
            "temp_factor": float(temp_factor),
            "milk_freq_factor": float(milk_freq_factor),
            "lactation_factor": float(lactation_factor)
        },
        "factor_contributions": {
            "feed_contribution_percent": float(feed_contribution),
            "health_contribution_percent": float(health_contribution),
            "temp_contribution_percent": float(temp_contribution),
            "milk_freq_contribution_percent": float(milk_freq_contribution),
            "lactation_contribution_percent": float(lactation_contribution)
        },
        "performance_analysis": {
            "main_limiting_factor": min_contribution_factor,
            "limiting_factor_contribution": float(min_contribution_value),
            "optimal_yield": float(optimal_yield),
            "improvement_potential": float(improvement_potential),
            "improvement_percentage": float(improvement_percentage),
            "current_efficiency": float(milk_yield / optimal_yield * 100) if optimal_yield > 0 else 0
        },
        "model_parameters": {
            "feed_quality": float(feed_quality),
            "health_status": float(health_status),
            "avg_temp": float(avg_temp),
            "milk_freq": float(milk_freq),
            "lactation_week": float(lactation_week)
        },
        "management_suggestions": suggestions,
        "optimal_conditions": {
            "optimal_feed_quality": 100.0,
            "optimal_health_status": 100.0,
            "optimal_temperature": 20.0,
            "optimal_milk_freq": 2.0,
            "peak_lactation_week": 30.0,
            "optimal_lactation_factor": float(optimal_lactation_factor)
        }
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
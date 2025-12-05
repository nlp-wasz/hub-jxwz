import numpy as np
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Real Estate Monthly Rent Prediction Model")

@mcp.tool()
def calculate_monthly_rent(
    area: float = 80.0,                # 面积 (平方米)
    location_score: float = 8.0,       # 位置评分 (0-10分)
    age: float = 5.0,                  # 房龄 (年)
    bedrooms: int = 2,                 # 卧室数量 (间)
    distance_to_subway: float = 500.0  # 到地铁距离 (米)
) -> dict:
    """
    房产月租金收益预测模型
    
    建模背景:
    在房地产投资与资产管理中，准确估算房产的潜在租金收益是评估投资回报和制定租赁策略的关键环节。
    影响租金水平的因素包括房产本身的物理属性（如面积、房龄、卧室数量）、地理位置（如地段评分）
    以及周边配套设施（如交通便利性）。为了提供一种直观且具备一定实用性的估算方法，构建了一个
    基于关键变量的代数模型，用于预测房产的预期月租金收益。该模型结合了线性与非线性关系，
    旨在反映现实中租金定价的主要驱动因素。
    
    建模公式:
    月租金的计算公式如下：
    
    \[\text{Monthly Rent} = 50 \times \text{Area} \times \left( \frac{\text{Location Score}}{10} \right) \times \left( 1 - \frac{\text{Age}}{30} \right) \times (1 + 0.1 \times \text{Bedrooms}) \times \left( \frac{1000}{1000 + \text{Distance to Subway}} \right)\]
    
    该模型通过加权各项影响因子，量化各因素对基础租金的调节作用，从而得出预期的月租金水平。
    模型参数的选择和结构设计基于经验判断与市场合理性，适用于初步估算和快速评估场景。
    
    参数说明:
    - area: 房产面积，单位：平方米 (m²)
    - location_score: 位置评分，评估房产所在地段的综合质量，范围：0-10分，10分为最优
    - age: 房龄，房屋建成后的年数，单位：年
    - bedrooms: 卧室数量，房产内卧室的间数，单位：间
    - distance_to_subway: 到最近地铁站的距离，衡量交通便利性，单位：米 (m)
    
    返回:
    - dict: 包含月租金、各因子贡献值、模型参数和分析建议的字典
    
    使用示例:
    >>> result = calculate_monthly_rent(
    ...     area=80.0,
    ...     location_score=8.0,
    ...     age=5.0,
    ...     bedrooms=2,
    ...     distance_to_subway=500.0
    ... )
    >>> print(f"预期月租金: {result['monthly_rent']:.2f} 元/月")
    """
    
    # 1. 参数验证
    if area <= 0:
        raise ValueError("房产面积必须为正数")
    if location_score < 0 or location_score > 10:
        raise ValueError("位置评分应在0-10之间")
    if age < 0:
        raise ValueError("房龄不能为负数")
    if bedrooms < 0:
        raise ValueError("卧室数量不能为负数")
    if distance_to_subway < 0:
        raise ValueError("到地铁距离不能为负数")
    
    # 2. 计算各调整因子
    # 基础租金因子
    base_rent_factor = 50.0
    
    # 面积因子
    area_factor = area
    
    # 位置评分因子
    location_factor = location_score / 10.0
    
    # 房龄折旧因子
    # 公式中的 (1 - Age/30) 可能导致负数，需进行处理
    age_factor = 1.0 - (age / 30.0)
    # 如果房龄超过30年，设折旧因子为最小值0.01（避免负租金）
    if age_factor < 0.01:
        age_factor = 0.01
    
    # 卧室数量因子
    bedrooms_factor = 1.0 + 0.1 * bedrooms
    
    # 地铁便利性因子
    subway_factor = 1000.0 / (1000.0 + distance_to_subway)
    # 确保分母不为零且因子在合理范围内
    if distance_to_subway < 0:
        subway_factor = 1.0  # 如果距离为负，设为最佳值
    
    # 3. 计算月租金
    monthly_rent = base_rent_factor * area_factor * location_factor * age_factor * bedrooms_factor * subway_factor
    
    # 4. 计算各因子的贡献度（相对于最优情况）
    # 最优情况下的因子值
    optimal_location_factor = 1.0  # 位置评分10分
    optimal_age_factor = 1.0  # 新房（房龄0年）
    optimal_bedrooms_factor = 1.0 + 0.1 * 3  # 假设3间卧室为最佳
    optimal_subway_factor = 1.0  # 地铁距离为0
    
    # 各因子的相对贡献（当前因子值 / 最优因子值）
    area_contribution = area_factor / 100.0 * 100  # 假设100平米为参考
    location_contribution = location_factor / optimal_location_factor * 100
    age_contribution = age_factor / optimal_age_factor * 100
    bedrooms_contribution = bedrooms_factor / optimal_bedrooms_factor * 100
    subway_contribution = subway_factor / optimal_subway_factor * 100
    
    # 5. 计算各因子对租金的绝对影响
    # 基准租金（仅考虑面积和基础租金）
    base_rent_only = base_rent_factor * area_factor
    
    # 计算每个因子单独调整后的租金
    rent_without_location = base_rent_only * 1.0 * age_factor * bedrooms_factor * subway_factor
    location_effect = monthly_rent - rent_without_location
    
    rent_without_age = base_rent_only * location_factor * 1.0 * bedrooms_factor * subway_factor
    age_effect = monthly_rent - rent_without_age
    
    rent_without_bedrooms = base_rent_only * location_factor * age_factor * 1.0 * subway_factor
    bedrooms_effect = monthly_rent - rent_without_bedrooms
    
    rent_without_subway = base_rent_only * location_factor * age_factor * bedrooms_factor * 1.0
    subway_effect = monthly_rent - rent_without_subway
    
    # 6. 计算租金单价（元/平米/月）
    if area > 0:
        rent_per_sqm = monthly_rent / area
    else:
        rent_per_sqm = 0
    
    # 7. 计算年化租金收益
    annual_rent = monthly_rent * 12
    
    # 8. 评估房产的租赁潜力
    # 基于租金单价评估
    if rent_per_sqm >= 100:
        rental_potential = "极高"
        potential_color = "green"
    elif rent_per_sqm >= 70:
        rental_potential = "高"
        potential_color = "light-green"
    elif rent_per_sqm >= 40:
        rental_potential = "中等"
        potential_color = "yellow"
    elif rent_per_sqm >= 20:
        rental_potential = "较低"
        potential_color = "orange"
    else:
        rental_potential = "低"
        potential_color = "red"
    
    # 9. 识别主要限制因素
    contributions = {
        "位置": location_contribution,
        "房龄": age_contribution,
        "卧室数量": bedrooms_contribution,
        "交通便利性": subway_contribution
    }
    
    min_contribution_factor = min(contributions, key=contributions.get)
    min_contribution_value = contributions[min_contribution_factor]
    
    # 10. 生成投资建议
    suggestions = []
    
    if location_contribution < 70:
        suggestions.append(f"位置评分较低 ({location_score:.1f}/10)，建议考虑地段改善或降低租金预期")
    
    if age_contribution < 60:
        suggestions.append(f"房龄较大 ({age:.1f}年)，建议考虑装修翻新以提升租金")
    
    if bedrooms_contribution < 80:
        suggestions.append(f"卧室数量较少 ({bedrooms}间)，对于家庭租客吸引力有限")
    
    if subway_contribution < 70:
        suggestions.append(f"交通便利性一般 (距地铁{distance_to_subway:.0f}米)，通勤租客可能减少")
    
    if rent_per_sqm < 30:
        suggestions.append(f"租金单价偏低 ({rent_per_sqm:.1f}元/㎡/月)，投资回报率可能较低")
    elif rent_per_sqm > 80:
        suggestions.append(f"租金单价较高 ({rent_per_sqm:.1f}元/㎡/月)，具备良好投资价值")
    
    # 计算潜在的租金提升空间
    optimal_rent = base_rent_factor * area_factor * optimal_location_factor * optimal_age_factor * optimal_bedrooms_factor * optimal_subway_factor
    improvement_potential = optimal_rent - monthly_rent
    improvement_percentage = (improvement_potential / monthly_rent) * 100 if monthly_rent > 0 else 0
    
    if improvement_percentage > 20:
        suggestions.append(f"通过改善条件，租金有{improvement_percentage:.1f}%的提升空间")
    
    if not suggestions:
        suggestions.append("当前房产条件良好，租金预测合理")
    
    return {
        "rent_prediction": {
            "monthly_rent": float(monthly_rent),
            "annual_rent": float(annual_rent),
            "rent_per_sqm": float(rent_per_sqm),
            "base_rent_only": float(base_rent_only),
            "rental_potential": rental_potential,
            "potential_color": potential_color
        },
        "factor_analysis": {
            "base_rent_factor": float(base_rent_factor),
            "area_factor": float(area_factor),
            "location_factor": float(location_factor),
            "age_factor": float(age_factor),
            "bedrooms_factor": float(bedrooms_factor),
            "subway_factor": float(subway_factor)
        },
        "factor_contributions": {
            "area_contribution_percent": float(area_contribution),
            "location_contribution_percent": float(location_contribution),
            "age_contribution_percent": float(age_contribution),
            "bedrooms_contribution_percent": float(bedrooms_contribution),
            "subway_contribution_percent": float(subway_contribution)
        },
        "factor_effects": {
            "location_effect": float(location_effect),
            "age_effect": float(age_effect),
            "bedrooms_effect": float(bedrooms_effect),
            "subway_effect": float(subway_effect)
        },
        "performance_analysis": {
            "main_limiting_factor": min_contribution_factor,
            "limiting_factor_contribution": float(min_contribution_value),
            "optimal_rent": float(optimal_rent),
            "improvement_potential": float(improvement_potential),
            "improvement_percentage": float(improvement_percentage)
        },
        "model_parameters": {
            "area": float(area),
            "location_score": float(location_score),
            "age": float(age),
            "bedrooms": bedrooms,
            "distance_to_subway": float(distance_to_subway)
        },
        "investment_recommendations": suggestions,
        "optimal_conditions": {
            "optimal_location_score": 10.0,
            "optimal_age": 0.0,
            "optimal_bedrooms": 3,
            "optimal_distance_to_subway": 0.0
        },
        "formula_explanation": {
            "formula": "Monthly Rent = 50 × Area × (Location Score/10) × (1 - Age/30) × (1 + 0.1 × Bedrooms) × (1000/(1000 + Distance to Subway))",
            "variables": {
                "Area": "房产面积 (m²)",
                "Location Score": "位置评分 (0-10)",
                "Age": "房龄 (年)",
                "Bedrooms": "卧室数量",
                "Distance to Subway": "到地铁距离 (m)"
            }
        }
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
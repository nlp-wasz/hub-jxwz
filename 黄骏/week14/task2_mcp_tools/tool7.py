import numpy as np
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Weight Change Prediction Model")

@mcp.tool()
def predict_weight_change(
    initial_weight: float = 70.0,      # 初始体重 (kg)
    daily_calorie_intake: float = 2500.0,  # 每日热量摄入 (kcal/天)
    daily_calorie_expenditure: float = 2000.0,  # 每日热量消耗 (kcal/天)
    conversion_coefficient: float = 7700.0,  # 转换系数 k (kcal/kg)
    num_days: int = 30                 # 预测天数 (天)
) -> dict:
    """
    体重变化预测模型
    
    建模背景:
    在健康管理与体重控制领域，理解并预测体重变化是一个重要的课题。
    个体的体重受多种因素影响，其中每日热量摄入与消耗之间的平衡是决定体重增减的关键因素之一。
    为了量化这一过程，可以构建一个基于差分方程的数学模型，用于模拟和预测个体在一定时间内的体重变化趋势。
    该模型有助于制定饮食与运动计划，辅助实现减重或增量目标。
    
    建模公式:
    模型采用一个简洁的差分方程形式，描述体重随时间变化的动态过程：
    
    \[W_{t+1} = W_t + \frac{C_t - E_t}{k}\]
    
    其中，$ W_t $ 表示第 $ t $ 天的体重，$ C_t $ 为当天的热量摄入量，
    $ E_t $ 为当天的热量消耗量，$ k $ 是热量与体重之间的转换系数。
    该模型假设体重变化与每日热量盈亏成线性关系，从而为体重管理提供了一个定量分析工具。
    
    参数说明:
    - initial_weight: 初始体重，单位：kg
    - daily_calorie_intake: 每日热量摄入，单位：kcal/天
    - daily_calorie_expenditure: 每日热量消耗，单位：kcal/天
    - conversion_coefficient: 转换系数 k，表示每公斤体重变化对应的热量变化，单位：kcal/kg
    - num_days: 预测天数，单位：天
    
    返回:
    - dict: 包含体重序列、变化量、关键指标和模型参数的字典
    
    使用示例:
    >>> result = predict_weight_change(
    ...     initial_weight=70.0,
    ...     daily_calorie_intake=2000.0,
    ...     daily_calorie_expenditure=2500.0,
    ...     conversion_coefficient=7700.0,
    ...     num_days=30
    ... )
    >>> print(f"最终体重: {result['final_weight']:.2f} kg")
    """
    
    # 1. 参数验证
    if initial_weight <= 0:
        raise ValueError("初始体重必须为正数")
    if daily_calorie_intake < 0:
        raise ValueError("每日热量摄入不能为负数")
    if daily_calorie_expenditure < 0:
        raise ValueError("每日热量消耗不能为负数")
    if conversion_coefficient <= 0:
        raise ValueError("转换系数必须为正数")
    if num_days <= 0:
        raise ValueError("预测天数必须为正数")
    
    # 2. 初始化体重序列
    weights = np.zeros(num_days + 1)  # +1是为了包含第0天（初始状态）
    weights[0] = initial_weight
    
    # 3. 计算每日热量平衡
    calorie_balance = daily_calorie_intake - daily_calorie_expenditure
    
    # 4. 使用差分方程迭代计算体重变化
    # 公式：W_{t+1} = W_t + (C_t - E_t) / k
    for t in range(num_days):
        weight_change = calorie_balance / conversion_coefficient
        weights[t + 1] = weights[t] + weight_change
        
        # 确保体重不会变为负数（虽然生理上不太可能，但作为保护）
        if weights[t + 1] < 0:
            weights[t + 1] = 0
            break  # 如果体重降为0，停止计算
    
    # 5. 提取时间序列
    days = np.arange(0, num_days + 1)
    
    # 6. 计算关键指标
    initial_w = weights[0]
    final_w = weights[-1]
    total_change = final_w - initial_w
    avg_daily_change = total_change / num_days if num_days > 0 else 0
    
    # 计算最大体重、最小体重
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    
    # 计算体重变化率
    if initial_w > 0:
        percentage_change = (total_change / initial_w) * 100
    else:
        percentage_change = 0
    
    # 计算达到目标体重所需天数（如果适用）
    # 假设目标体重变化为减重5kg或增重5kg
    target_weight_change = -5.0  # 默认减重5kg为目标
    target_weight = initial_w + target_weight_change
    
    days_to_target = None
    if total_change != 0:  # 避免除以零
        # 线性估算达到目标所需天数
        required_days = abs(target_weight_change / avg_daily_change) if avg_daily_change != 0 else None
        if required_days is not None:
            days_to_target = int(np.ceil(required_days))
    
    # 7. 评估体重变化趋势
    if total_change < -1.0:  # 减重超过1kg
        trend = "减重"
        trend_description = "体重呈下降趋势"
    elif total_change > 1.0:  # 增重超过1kg
        trend = "增重"
        trend_description = "体重呈上升趋势"
    else:  # 变化在±1kg内
        trend = "稳定"
        trend_description = "体重基本保持稳定"
    
    # 8. 计算热量赤字/盈余
    total_calorie_balance = calorie_balance * num_days
    estimated_weight_change_from_calories = total_calorie_balance / conversion_coefficient
    
    # 9. 生成健康建议
    suggestions = []
    
    if calorie_balance < 0:  # 热量赤字
        if abs(total_change) > 0.5:  # 有明显变化
            suggestions.append(f"每日热量赤字为 {abs(calorie_balance):.0f} kcal，预计每周减重约 {abs(avg_daily_change)*7:.2f} kg")
        else:
            suggestions.append("热量赤字较小，体重变化不明显")
    elif calorie_balance > 0:  # 热量盈余
        if abs(total_change) > 0.5:  # 有明显变化
            suggestions.append(f"每日热量盈余为 {calorie_balance:.0f} kcal，预计每周增重约 {avg_daily_change*7:.2f} kg")
        else:
            suggestions.append("热量盈余较小，体重变化不明显")
    else:  # 热量平衡
        suggestions.append("热量摄入与消耗平衡，体重应保持稳定")
    
    # 添加基于BMI的健康建议
    # 假设标准身高为1.7米来计算BMI
    standard_height = 1.70  # 米
    bmi_initial = initial_w / (standard_height ** 2)
    bmi_final = final_w / (standard_height ** 2)
    
    bmi_category_initial = categorize_bmi(bmi_initial)
    bmi_category_final = categorize_bmi(bmi_final)
    
    if bmi_category_initial != bmi_category_final:
        suggestions.append(f"BMI分类将从 '{bmi_category_initial}' 变为 '{bmi_category_final}'")
    
    return {
        "time_series": {
            "days": days.tolist(),
            "weights": weights.tolist()
        },
        "key_metrics": {
            "initial_weight": float(initial_w),
            "final_weight": float(final_w),
            "total_weight_change": float(total_change),
            "average_daily_change": float(avg_daily_change),
            "percentage_change": float(percentage_change),
            "maximum_weight": float(max_weight),
            "minimum_weight": float(min_weight),
            "weight_trend": trend,
            "trend_description": trend_description
        },
        "calorie_analysis": {
            "daily_calorie_balance": float(calorie_balance),
            "total_calorie_balance": float(total_calorie_balance),
            "conversion_coefficient": float(conversion_coefficient),
            "estimated_weight_change": float(estimated_weight_change_from_calories),
            "actual_weight_change": float(total_change),
            "calorie_discrepancy": float(total_change * conversion_coefficient - total_calorie_balance)
        },
        "target_analysis": {
            "target_weight_change": float(target_weight_change),
            "target_weight": float(target_weight),
            "days_to_reach_target": days_to_target,
            "current_rate_days_to_target": days_to_target,
            "required_daily_calorie_adjustment": (
                target_weight_change * conversion_coefficient / num_days if num_days > 0 else 0
            )
        },
        "bmi_analysis": {
            "bmi_initial": float(bmi_initial),
            "bmi_final": float(bmi_final),
            "bmi_category_initial": bmi_category_initial,
            "bmi_category_final": bmi_category_final,
            "standard_height_assumed": float(standard_height)
        },
        "model_parameters": {
            "initial_weight": float(initial_weight),
            "daily_calorie_intake": float(daily_calorie_intake),
            "daily_calorie_expenditure": float(daily_calorie_expenditure),
            "conversion_coefficient": float(conversion_coefficient),
            "num_days": num_days
        },
        "health_recommendations": suggestions,
        "formula_explanation": {
            "difference_equation": "W_{t+1} = W_t + (C_t - E_t) / k",
            "variables": {
                "W_t": "第t天的体重 (kg)",
                "C_t": "第t天的热量摄入 (kcal)",
                "E_t": "第t天的热量消耗 (kcal)",
                "k": "热量与体重转换系数 (kcal/kg)"
            }
        }
    }

def categorize_bmi(bmi: float) -> str:
    """
    根据BMI值分类
    
    参数:
    - bmi: 身体质量指数
    
    返回:
    - BMI分类字符串
    """
    if bmi < 18.5:
        return "体重过轻"
    elif 18.5 <= bmi < 24:
        return "正常范围"
    elif 24 <= bmi < 28:
        return "超重"
    else:
        return "肥胖"

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
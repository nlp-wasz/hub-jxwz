import numpy as np
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Agricultural Product Retail Revenue Model")

@mcp.tool()
def simulate_retail_revenue(
    input_kg: float = 1000.0,           # 进货量 (kg)
    base_price: float = 10.0,           # 基准价格 (元/kg)
    fluctuation_range: float = 2.0,     # 价格波动范围 (元/kg)
    random_seed: int = None             # 随机种子 (确保可复现性)
) -> dict:
    """
    农产品零售销售额模拟模型
    
    建模背景:
    在农产品批发与零售业务中，零售价格往往受到市场供需、季节性变化以及外部经济因素的
    影响而产生波动。为了更贴近实际市场行为，构建一个具备随机性特征的价格模型，能够
    反映零售收入在不同进货量下的不确定性。该模型基于一个基准价格，并引入价格浮动机制，
    从而模拟真实业务场景中销售额的变化情况。
    
    建模公式:
    零售销售额通过以下关系进行建模：
    
    \[Revenue = \text{Input\_kg} \times (\text{Base\_price} + \text{Fluctuation})\]
    
    其中：
    - Revenue 表示预期的零售销售额；
    - Input_kg 表示农产品的进货量（单位：千克）；
    - Base_price 为设定的基准价格（单位：元/千克）；
    - Fluctuation 是一个在固定区间内均匀分布的随机变量，表示市场价格的波动幅度。
    
    该模型通过控制随机种子确保模拟结果的可复现性，同时保留价格波动带来的不确定性特征。
    
    参数说明:
    - input_kg: 进货量，表示农产品的进货数量，单位：千克 (kg)
    - base_price: 基准价格，表示农产品的预期销售价格，单位：元/千克 (RMB/kg)
    - fluctuation_range: 价格波动范围，表示市场价格在基准价格上下波动的最大幅度，单位：元/千克 (RMB/kg)
    - random_seed: 随机种子，用于确保模拟结果的可复现性。如果为None，则使用系统随机生成
    
    返回:
    - dict: 包含模拟结果、价格信息和模型参数的字典
    
    使用示例:
    >>> result = simulate_retail_revenue(
    ...     input_kg=1000.0,
    ...     base_price=10.0,
    ...     fluctuation_range=2.0,
    ...     random_seed=42
    ... )
    >>> print(f"零售销售额: {result['revenue']:.2f} 元")
    """
    
    # 1. 参数验证
    if input_kg <= 0:
        raise ValueError("进货量必须为正数")
    if base_price <= 0:
        raise ValueError("基准价格必须为正数")
    if fluctuation_range < 0:
        raise ValueError("价格波动范围不能为负数")
    
    # 2. 设置随机种子（确保结果可复现）
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 3. 生成价格波动值（在[-fluctuation_range, fluctuation_range]范围内均匀分布）
    fluctuation = np.random.uniform(
        low=-fluctuation_range,
        high=fluctuation_range
    )
    
    # 4. 计算实际销售价格
    actual_price = base_price + fluctuation
    
    # 确保实际价格不会为负数
    if actual_price < 0:
        actual_price = 0.0
        fluctuation = -base_price  # 调整波动值，使实际价格不为负
    
    # 5. 计算零售销售额
    revenue = input_kg * actual_price
    
    # 6. 计算基准收入（无波动情况下的收入）
    baseline_revenue = input_kg * base_price
    
    # 7. 计算收入变化量
    revenue_change = revenue - baseline_revenue
    revenue_change_percentage = (revenue_change / baseline_revenue) * 100 if baseline_revenue > 0 else 0
    
    # 8. 分析价格分布特征（模拟1000次，用于统计分析）
    np.random.seed(random_seed if random_seed is not None else np.random.randint(0, 10000))
    sample_size = 1000
    price_samples = base_price + np.random.uniform(
        low=-fluctuation_range,
        high=fluctuation_range,
        size=sample_size
    )
    
    # 确保价格样本不为负
    price_samples = np.maximum(price_samples, 0)
    
    # 计算价格统计
    avg_price = np.mean(price_samples)
    min_price = np.min(price_samples)
    max_price = np.max(price_samples)
    price_std = np.std(price_samples)
    
    # 9. 计算对应收入的统计
    revenue_samples = input_kg * price_samples
    avg_revenue = np.mean(revenue_samples)
    min_revenue = np.min(revenue_samples)
    max_revenue = np.max(revenue_samples)
    revenue_std = np.std(revenue_samples)
    
    # 10. 计算风险指标
    # 风险定义为收入低于基准收入90%的概率
    revenue_threshold = baseline_revenue * 0.9
    downside_risk = np.sum(revenue_samples < revenue_threshold) / sample_size * 100
    
    # 计算波动带来的潜在损失/收益
    potential_loss = baseline_revenue - min_revenue
    potential_gain = max_revenue - baseline_revenue
    
    # 11. 生成经营建议
    suggestions = []
    
    if fluctuation_range == 0:
        suggestions.append("价格无波动，收入完全可预测")
    else:
        # 计算波动对收入的影响程度
        volatility_impact = revenue_std / avg_revenue * 100 if avg_revenue > 0 else 0
        
        if volatility_impact < 5:
            suggestions.append("价格波动较小，收入相对稳定")
        elif volatility_impact < 15:
            suggestions.append("价格波动适中，需关注市场变化")
        else:
            suggestions.append("价格波动较大，建议采取套期保值或期货等风险管理措施")
    
    if downside_risk > 20:
        suggestions.append(f"收入下降风险较高 ({downside_risk:.1f}%概率)，建议谨慎进货")
    elif downside_risk > 5:
        suggestions.append(f"收入有一定下降风险 ({downside_risk:.1f}%概率)，建议适当控制库存")
    else:
        suggestions.append(f"收入下降风险较低 ({downside_risk:.1f}%概率)")
    
    # 根据波动范围提供价格策略建议
    if fluctuation_range / base_price > 0.3:
        suggestions.append("市场价格波动剧烈，建议设置价格区间进行销售")
    elif fluctuation_range / base_price > 0.1:
        suggestions.append("市场价格波动明显，建议定期调整销售价格")
    
    # 12. 评估本次模拟结果
    if revenue > baseline_revenue:
        result_evaluation = "有利"
        evaluation_color = "green"
    elif revenue < baseline_revenue:
        result_evaluation = "不利"
        evaluation_color = "red"
    else:
        result_evaluation = "中性"
        evaluation_color = "gray"
    
    return {
        "simulation_result": {
            "actual_price": float(actual_price),
            "price_fluctuation": float(fluctuation),
            "revenue": float(revenue),
            "baseline_revenue": float(baseline_revenue),
            "revenue_change": float(revenue_change),
            "revenue_change_percentage": float(revenue_change_percentage),
            "result_evaluation": result_evaluation,
            "evaluation_color": evaluation_color
        },
        "price_analysis": {
            "average_price": float(avg_price),
            "minimum_price": float(min_price),
            "maximum_price": float(max_price),
            "price_standard_deviation": float(price_std),
            "price_volatility": float(price_std / avg_price * 100) if avg_price > 0 else 0,
            "price_range": float(max_price - min_price)
        },
        "revenue_analysis": {
            "average_revenue": float(avg_revenue),
            "minimum_revenue": float(min_revenue),
            "maximum_revenue": float(max_revenue),
            "revenue_standard_deviation": float(revenue_std),
            "revenue_volatility": float(revenue_std / avg_revenue * 100) if avg_revenue > 0 else 0,
            "downside_risk_percentage": float(downside_risk),
            "potential_loss": float(potential_loss),
            "potential_gain": float(potential_gain),
            "risk_reward_ratio": float(potential_gain / potential_loss) if potential_loss > 0 else float('inf')
        },
        "model_parameters": {
            "input_kg": float(input_kg),
            "base_price": float(base_price),
            "fluctuation_range": float(fluctuation_range),
            "random_seed": random_seed,
            "sample_size": sample_size
        },
        "business_recommendations": suggestions,
        "formula_explanation": {
            "revenue_formula": "Revenue = Input_kg × (Base_price + Fluctuation)",
            "fluctuation_distribution": f"Fluctuation ~ Uniform(-{fluctuation_range}, {fluctuation_range})",
            "variables": {
                "Input_kg": "进货量 (kg)",
                "Base_price": "基准价格 (元/kg)",
                "Fluctuation": "价格波动值 (元/kg)",
                "Revenue": "零售销售额 (元)"
            }
        }
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
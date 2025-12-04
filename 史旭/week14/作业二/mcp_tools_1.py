# 将解析的pdf文档内容，转换为可计算的 mcp 工具
import math
from fastmcp import FastMCP
from typing import Union, List

mcp = FastMCP(name="MCP 工具1")


# 工具 1
@mcp.tool()
def calculate_dissolved_oxygen(
        a: float,
        b: float,
        c: float,
        d: float,
        t: Union[float, List[float]]
) -> Union[float, List[float]]:
    """
    计算水产养殖系统中溶解氧浓度 DO(t) = a * exp(-b*t) + c * sin(d*t)

    参数:
        a (float): 初始溶解氧释放量（mg/L）
        b (float): 衰减系数（1/hour）
        c (float): 环境扰动振幅（mg/L）
        d (float): 扰动频率（rad/hour）
        t (float or list of float): 时间点（小时）

    返回:
        float 或 list: 对应时间点的 DO 浓度（mg/L）
    """
    if isinstance(t, list):
        return [a * math.exp(-b * ti) + c * math.sin(d * ti) for ti in t]
    else:
        return a * math.exp(-b * t) + c * math.sin(d * t)


# 工具 2
@mcp.tool()
def predict_daily_orders(
        ad_spend: float,
        discount_rate: float,
        prev_orders: int
) -> int:
    """
    基于线性差分方程预测电商当日订单数量。

    公式：orders_t = 0.05 * ad_spend + 100 * discount_rate + 0.7 * prev_orders

    参数:
        ad_spend (float): 当日广告支出（单位：元）
        discount_rate (float): 当日折扣力度（0.0 ~ 1.0，例如 0.2 表示打 8 折）
        prev_orders (int): 前一日订单数量（整数）

    返回:
        int: 预测的当日订单数量（向下取整，订单数为整数）
    """
    alpha = 0.05
    beta = 100.0
    gamma = 0.7

    orders_t = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
    return max(0, int(orders_t))  # 订单数不能为负


# 工具 3
@mcp.tool()
def predict_crop_yield(
        temp: float,
        rainfall: float,
        fertilizer: float,
        sunlight: float,
        soil_quality: float
) -> float:
    """
    基于环境与管理因素预测农作物产量（吨/公顷）。

    公式：
        yield = 5.0
                * temp_factor
                * rainfall_factor
                * fertilizer_factor
                * sunlight_factor
                * soil_factor

    各因子定义：
      - temp_factor = max(0, 1 - |temp - 25| / 25)
      - rainfall_factor = max(0, 1 - |rainfall - 600| / 600)
      - fertilizer_factor = 1 + fertilizer / 200
      - sunlight_factor = clamp(0.8 + (sunlight / 12) * 0.4, 0.8, 1.2)
      - soil_factor = 1 + soil_quality

    参数:
        temp (float): 平均生长温度（℃）
        rainfall (float): 生长期间降水量（mm）
        fertilizer (float): 施肥量（kg/公顷）
        sunlight (float): 每日平均光照时长（小时）
        soil_quality (float): 土壤质量指数（无量纲，通常 ≥0）

    返回:
        float: 预测产量（吨/公顷），最小为 0.0
    """
    base_yield = 5.0

    # 温度因子（最适 25℃）
    temp_factor = max(0.0, 1.0 - abs(temp - 25.0) / 25.0)

    # 降水因子（最适 600mm）
    rainfall_factor = max(0.0, 1.0 - abs(rainfall - 600.0) / 600.0)

    # 施肥因子（正相关）
    fertilizer_factor = 1.0 + fertilizer / 200.0

    # 光照因子（8~12 小时最优 → 因子 0.8~1.2）
    sunlight_factor_raw = 0.8 + (sunlight / 12.0) * 0.4
    sunlight_factor = max(0.8, min(1.2, sunlight_factor_raw))

    # 土壤因子
    soil_factor = 1.0 + soil_quality

    # 计算产量
    yield_pred = (
            base_yield *
            temp_factor *
            rainfall_factor *
            fertilizer_factor *
            sunlight_factor *
            soil_factor
    )

    return max(0.0, yield_pred)


# 工具 4
@mcp.tool()
def predict_crop_yield_deterministic(
        F: float,
        I: float,
        T: float
) -> float:
    """
    基于确定性方程预测单位面积作物产量（kg/ha）。

    公式：Y = a * F + b * I - c * T^2

    默认系数（可根据实际校准）：
      - a = 50.0  # 土壤肥力贡献 (kg/ha per unit)
      - b = 10.0  # 灌溉贡献 (kg/ha per mm/week)
      - c = 0.5   # 温度抑制系数 (kg/ha per ℃²)

    参数:
        F (float): 土壤肥力指数（无量纲，例如 3.2）
        I (float): 每周灌溉量（mm/week，例如 40.0）
        T (float): 平均气温（℃，例如 22.5）

    返回:
        float: 预测产量 Y（kg/ha），最小为 0.0（产量不能为负）
    """
    # 经验系数（可配置）
    a = 50.0
    b = 10.0
    c = 0.5

    Y = a * F + b * I - c * (T ** 2)
    return max(0.0, Y)  # 产量不能为负


# 工具 5
@mcp.tool()
def calculate_drying_process(
        M0: float,
        k: float,
        T: float
) -> dict:
    """
    计算食品干燥过程中的剩余水分和累计蒸发量。

    公式：
      - 剩余水分: M(T) = M0 * exp(-k * T)
      - 累计蒸发量: Evap(T) = M0 * (T + exp(-k*T)/k - 1)

    参数:
        M0 (float): 初始水分含量（单位：g 或 kg）
        k (float): 水分蒸发速率常数（单位：1/时间单位，如 1/hour）
        T (float): 干燥时间（单位：与 k 的时间单位一致，如 hour）

    返回:
        dict: {
            "remaining_moisture": float,   # M(T)
            "evaporated_amount": float,    # Evap(T)
            "evaporation_ratio": float     # 蒸发比例（0~1）
        }
    """
    if k <= 0:
        raise ValueError("蒸发速率常数 k 必须大于 0")
    if T < 0:
        raise ValueError("干燥时间 T 不能为负")
    if M0 < 0:
        raise ValueError("初始水分 M0 不能为负")

    # 剩余水分
    remaining = M0 * math.exp(-k * T)

    # 累计蒸发量（注意：当 k 很小时，exp(-kT)/k 可能数值不稳定，但一般 k>0.01 无问题）
    evaporated = M0 * (T + math.exp(-k * T) / k - 1)

    # 确保物理合理性：蒸发量 ∈ [0, M0 * T] 且 剩余 ≥ 0
    evaporated = max(0.0, min(evaporated, M0 * T))
    remaining = max(0.0, remaining)

    # 蒸发比例（相对于理论最大可能蒸发？这里用累计蒸发 / (M0 * T) 不合适）
    # 更合理：已蒸发质量占“若全程以初始速率蒸发”的比例？但通常直接返回绝对值即可
    # 此处提供“已损失水分占初始水分的比例”仅适用于 T→∞ 极限，不推荐
    # 所以我们只返回绝对量

    return {
        "remaining_moisture": round(remaining, 4),
        "evaporated_amount": round(evaporated, 4)
    }

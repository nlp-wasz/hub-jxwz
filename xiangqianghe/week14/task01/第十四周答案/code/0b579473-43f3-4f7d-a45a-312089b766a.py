
import math  # 必须保留
import numpy as np  # 必须保留
from fastmcp import FastMCP  # 必须保留
from typing import Annotated, Union  # 必须保留
mcp = FastMCP(name="food_drying_model")  # 必须保留

@mcp.tool
def calculate_evaporated_moisture(
    M0: Annotated[float, "初始水分含量，表示食品在干燥开始时刻的水分质量或比例"],
    k: Annotated[float, "水分蒸发速率常数，反映干燥过程中水分减少的速度"],
    T: Annotated[float, "干燥时间，表示从干燥开始到当前时刻的时间长度"]
) -> Annotated[float, "在时间区间 [0, T] 内的累计水分蒸发量，表示食品在此期间损失的总水分"]:
    """
    在食品加工与制造过程中，干燥是一个关键的工艺环节，广泛应用于食品保存、品质控制及延长货架期等方面。
    水分含量的控制对于确保食品的稳定性和安全性至关重要。该函数基于水分蒸发速率与当前水分含量成正比的假设，
    利用指数衰减模型描述食品在干燥过程中水分随时间的变化，并通过积分方法计算在给定时间范围内食品的累计水分蒸发量，
    为干燥工艺的设计与优化提供理论支持。

    参数说明:
    - M0 (float): 初始水分含量，单位可为克或质量分数，表示干燥起始时的水分值。
    - k (float): 水分蒸发速率常数，单位通常为1/时间（如1/h），决定干燥速度。
    - T (float): 干燥时间，单位为小时或分钟，需与k的单位一致。

    返回值:
    - float: 在时间区间 [0, T] 内的累计蒸发水分总量，单位与M0一致。

    模型公式:
    假设瞬时水分含量为 M(t) = M0 * exp(-k * t)
    瞬时蒸发速率为 dM/dt = -k * M0 * exp(-k * t)
    累计蒸发量 = ∫₀ᵀ [M0 - M(t)] dt = ∫₀ᵀ M0*(1 - exp(-k*t)) dt
    解得：evaporated = M0 * (T + (exp(-k*T) - 1)/k)  （注意：原sympy表达式有误，此处已修正）

    注意：原提供的Sympy代码中对 expr = M0*(1-exp(-k*T)) 进行关于 t 的积分是不正确的，
    因为 expr 不应直接使用 T，而应是关于时间变量 t 的函数。正确推导如下：
        cumulative_evaporation = ∫₀ᵀ M0*(1 - exp(-k*t)) dt
                              = M0 * [t + (1/k)*exp(-k*t)] from 0 to T
                              = M0 * (T + (1/k)*(exp(-k*T) - 1))
    """
    if M0 < 0 or k <= 0 or T < 0:
        raise ValueError("M0 must be non-negative, k must be positive, and T must be non-negative.")

    # 正确的积分结果：∫₀ᵀ M0*(1 - exp(-k*t)) dt
    try:
        evaporated = M0 * (T + (math.exp(-k * T) - 1) / k)
        return evaporated
    except OverflowError:
        return float('inf')

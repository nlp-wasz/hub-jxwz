
import math  # 必须保留
import numpy as np  # 必须保留
from fastmcp import FastMCP  # 必须保留
from typing import Annotated, Union  # 必须保留
mcp = FastMCP(name="ecommerce_forecasting")  # 必须保留

@mcp.tool
def predict_orders(
    ad_spend: Annotated[float, "当日广告支出金额，单位为元"],
    discount_rate: Annotated[float, "当日折扣率，取值范围0到1之间的浮点数"],
    prev_orders: Annotated[int, "前一日的订单数量"]
) -> Annotated[float, "预测的当日订单数量，为一个实数"]:
    """
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有重要意义。
    该模型基于广告支出、当日折扣力度以及前一日订单数量三个关键驱动因素，
    构建了一个一阶线性差分方程模型，用于预测当日订单数量。
    模型具有良好的可解释性和短期预测能力，适用于趋势模拟与敏感性分析。

    模型公式：
        orders_t = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
    其中：
        alpha = 0.05   # 广告支出对订单的边际效应
        beta = 100     # 折扣力度对订单的基础激励系数
        gamma = 0.7    # 前一日订单的自回归系数，反映订单持续性

    参数说明：
        - ad_spend (float): 当日广告支出金额（单位：元），应为非负数。
        - discount_rate (float): 当日折扣率，取值范围 [0, 1] 的浮点数（例如：0.2 表示八折）。
        - prev_orders (int): 前一日实际完成的订单数量，应为非负整数。

    返回值：
        float: 预测的当日订单数量，为一个非负实数（可根据业务需要四舍五入或取整）。

    示例：
        predict_orders(ad_spend=2000, discount_rate=0.3, prev_orders=500)
        => 0.05*2000 + 100*0.3 + 0.7*500 = 100 + 30 + 350 = 480.0
    """
    # 定义模型参数
    alpha = 0.05
    beta = 100
    gamma = 0.7

    # 计算预测订单量
    predicted_orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders

    return float(predicted_orders)

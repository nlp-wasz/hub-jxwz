
import math  # 必须保留
import numpy as np  # 必须保留
from fastmcp import FastMCP  # 必须保留
from typing import Annotated, Union  # 必须保留
mcp = FastMCP(name="agriculture_yield_predictor")  # 必须保留

@mcp.tool
def predict_crop_yield(
    F: Annotated[float, "土壤肥力指数"],
    I: Annotated[float, "每周灌溉量 (mm/week)"],
    T: Annotated[float, "平均气温 (℃)"],
    a: Annotated[float, "土壤肥力对产量的贡献系数"],
    b: Annotated[float, "灌溉量对产量的贡献系数"],
    c: Annotated[float, "温度平方项的抑制系数"]
):
    """
    在农业科研领域,准确预测作物产量对于制定种植策略、优化资源配置以及提升农业生产效率具有重要意义。
    本函数基于关键环境与土壤因素，使用确定性模型估算单位面积上的作物产量。
    模型综合考虑了土壤肥力、灌溉量以及气温对作物生长的影响，适用于在可控环境条件下进行定量分析和趋势预测。
    
    产量计算公式为：Y = a * F + b * I - c * T^2
    其中：
      - a * F 表示土壤肥力带来的正向增产效应
      - b * I 表示灌溉量带来的正向增产效应
      - c * T^2 表示高温（或低温）超出适宜范围后对产量的负向抑制效应
    
    参数说明：
      F (float): 土壤肥力指数，数值越高表示土壤养分越丰富。
      I (float): 每周灌溉量，单位为毫米每星期 (mm/week)。
      T (float): 平均气温，单位为摄氏度 (℃)。
      a (float): 土壤肥力对产量的贡献系数，反映单位肥力提升带来的产量增加。
      b (float): 灌溉量对产量的贡献系数，反映单位灌溉量提升带来的产量增加。
      c (float): 温度平方项的抑制系数，用于模拟极端温度对作物生长的不利影响。

    返回值：
      Y (float): 单位面积作物产量，单位为千克每公顷 (kg/ha)。
    """
    Y = a * F + b * I - c * T**2
    return float(Y)

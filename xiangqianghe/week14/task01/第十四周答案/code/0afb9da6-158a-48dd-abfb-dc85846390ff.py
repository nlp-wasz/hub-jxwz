
import math  # 必须保留
import numpy as np  # 必须保留
from fastmcp import FastMCP  # 必须保留
from typing import Annotated, Union  # 必须保留
from sympy import Abs, Max
mcp = FastMCP(name="agriculture_model")  # 必须保留

@mcp.tool
def predict_crop_yield(
    temp: Annotated[float, "平均生长温度（单位：℃）"],
    rainfall: Annotated[float, "生长期间降水量（单位：mm）"],
    fertilizer: Annotated[float, "施肥量（单位：kg/ha）"],
    sunlight: Annotated[float, "每日平均光照时长（单位：小时）"],
    soil_quality: Annotated[float, "土壤质量指数（无量纲，数值越高表示质量越好）"]
) -> Annotated[dict, "{'yield_prediction': '预测的作物产量（单位：吨/公顷），非负值'}"]:
    """
    该模型用于预测作物产量，综合考虑平均生长温度、生长期间降水量、施肥量、每日平均光照时长和土壤质量指数五个关键因素的影响。
    通过归一化处理与参数组合，反映不同环境与管理条件下作物产量的变化趋势，为农业决策提供理论支持。

    参数说明：
    - temp (float): 平均生长温度，单位为摄氏度（℃）。最适温度为25℃，偏离此值将降低产量潜力。
    - rainfall (float): 生长期间总降水量，单位为毫米（mm）。理想降水量为600mm，过多或过少都会影响产量。
    - fertilizer (float): 施肥量，单位为千克每公顷（kg/ha）。适量施肥可线性提升产量。
    - sunlight (float): 每日平均光照时长，单位为小时。光照越长（最多12小时），光合作用越强，产量越高。
    - soil_quality (float): 土壤质量指数，无量纲，数值越高表示土壤肥力、结构等条件越好，正向影响产量。

    函数逻辑：
    基础产量为5.0吨/公顷，各因素通过乘性因子共同作用：
    - 温度因子：以25℃为最优，偏离则减产，使用绝对值归一化。
    - 降水因子：以600mm为最优，偏离则减产。
    - 施肥因子：每增加200kg/ha，产量提升100%。
    - 光照因子：在8~12小时范围内按比例增强（0.8到1.2倍）。
    - 土壤因子：直接加成，如土壤质量指数为0.5，则增产50%。

    最终产量不会为负，低于0时返回0。
    """
    base_yield = 5.0
    temp_factor = 1.0 - Abs(temp - 25) / 25
    rainfall_factor = 1.0 - Abs(rainfall - 600) / 600
    fertilizer_factor = 1.0 + fertilizer / 200
    sunlight_factor = 0.8 + (sunlight / 12) * 0.4
    soil_factor = 1.0 + soil_quality

    yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor
    result = Max(0, yield_prediction)

    return {"yield_prediction": float(result)}

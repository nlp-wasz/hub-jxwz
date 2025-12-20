# 解析10个pdf文档，并定义10个对应的mcp tool
## 提示词设计
解析上述文档，生成计算公式，并根据公式生成对应的python函数，可以使用sympy，也可以使用numpy，
生成函数后部署为mcp tool，注意mcp tool尽量使用typing Annotated注解清楚,只输出代码不要输出其他
## 解析思路
1. 将pdf转为图像
2. 利用qwen-vl进行解析和函数生成
3. 利用qwen3生成函数用例测试

## 解析过程
示例一：0a948fc4-b083-44c6-af02-70be51108f7.pdf
函数结果：

```python
import numpy as np
from typing import Annotated

def calculate_do(
    t: Annotated[float, "时间（单位：小时或天）"],
    a: Annotated[float, "初始溶解氧释放量（初始氧含量）"],
    b: Annotated[float, "溶解氧衰减系数（自然下降速率）"],
    c: Annotated[float, "环境扰动振幅（周期性影响强度）"],
    d: Annotated[float, "环境扰动频率（周期快慢）"]
) -> Annotated[float, "计算得到的溶解氧浓度 DO(t)"]:
    """
    根据非线性动力学模型计算溶解氧浓度随时间的变化。
    
    公式：DO(t) = a * exp(-b * t) + c * sin(d * t)
    
    参数:
        t: 时间点
        a: 初始氧含量
        b: 衰减系数
        c: 扰动振幅
        d: 扰动频率
    
    返回:
        DO(t): 在时间 t 的溶解氧浓度
    """
    return a * np.exp(-b * t) + c * np.sin(d * t)
```

示例二：0a96883f-a7ee-4aa7-b020-efdb1f634d0.pdf
```python
import numpy as np
from typing import List, Annotated

def predict_orders_mcp(
    ad_spend: Annotated[List[float], "当日广告支出列表（单位：元）"],
    discount_rate: Annotated[List[float], "当日折扣力度列表（如 0.8 表示 8 折）"],
    prev_orders: Annotated[List[float], "前一日订单数量列表"]
) -> Annotated[List[float], "预测的当日订单数量列表"]:
    """
    MCP Tool：基于线性差分方程模型预测当日订单数量。
    
    公式：orders_t = 0.05 * ad_spend_t + 100 * discount_rate_t + 0.7 * prev_orders_t
    
    参数:
        ad_spend: 当日广告支出序列
        discount_rate: 当日折扣力度序列（0~1）
        prev_orders: 前一日订单数量序列
    
    返回:
        预测的当日订单数量列表，与输入长度一致
    """
    # 转换为 NumPy 数组以便向量化计算
    ad_spend_arr = np.array(ad_spend)
    discount_rate_arr = np.array(discount_rate)
    prev_orders_arr = np.array(prev_orders)
    
    # 应用线性模型
    orders_pred = (
        0.05 * ad_spend_arr +
        100 * discount_rate_arr +
        0.7 * prev_orders_arr
    )
    
    return orders_pred.tolist()

```

示例三：00ac792a-04dd-4639-abbd-d7f78cbb7ea.pdf

```python
import numpy as np
from typing import List, Annotated

def simulate_interaction_mcp(
    x: Annotated[List[float], "输入变量 x 的值列表（如环境因子或信号强度）"],
    y: Annotated[List[float], "输入变量 y 的值列表（如另一环境因子或耦合信号）"]
) -> Annotated[List[float], "综合输出值 fun(x, y) 列表"]:
    """
    MCP Tool：模拟两个输入变量 x 和 y 的非线性交互作用对系统输出的影响。
    
    公式：fun(x, y) = 2.5 * sin(x) + 1.8 * cos(y) + 0.3 * x * y
    
    参数:
        x: 输入变量 x 的序列（可为角度或归一化值）
        y: 输入变量 y 的序列
    
    返回:
        对应每个 (x_i, y_i) 的输出值列表
    """
    # 转换为 NumPy 数组以便向量化计算
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # 应用非线性交互模型
    output = (
        2.5 * np.sin(x_arr) +
        1.8 * np.cos(y_arr) +
        0.3 * x_arr * y_arr
    )
    
    return output.tolist()
```

示例四：0afb9da6-158a-48dd-abfb-dc85846390ff.md

```python
import numpy as np
from typing import List, Annotated

def predict_agricultural_yield_mcp(
    temp: Annotated[List[float], "平均生长温度（单位：℃）"],
    rainfall: Annotated[List[float], "生长期间降水量（单位：mm）"],
    fertilizer: Annotated[List[float], "施肥量（单位：kg/ha 或自定义单位）"],
    sunlight: Annotated[List[float], "每日平均光照时长（单位：小时）"],
    soil_quality: Annotated[List[float], "土壤质量指数（建议范围：0.0 ~ 1.0）"]
) -> Annotated[List[float], "预测作物产量（单位：吨/公顷），非负值"]:
    """
    MCP Tool：基于多因子代数模型预测农业作物产量。
    
    模型综合温度、降水、施肥、光照、土壤质量五大因素，通过乘法耦合方式计算预测产量。
    所有因子均归一化处理，确保结果合理且可解释。
    
    参数:
        temp: 平均生长温度列表
        rainfall: 生长季总降水量列表
        fertilizer: 施肥量列表
        sunlight: 日均光照时长列表（超过12小时按12计算）
        soil_quality: 土壤质量指数列表（推荐0~1）
    
    返回:
        预测产量列表（吨/公顷），自动截断负值为0
    """
    # 转换为 NumPy 数组以便向量化计算
    temp_arr = np.array(temp)
    rainfall_arr = np.array(rainfall)
    fertilizer_arr = np.array(fertilizer)
    sunlight_arr = np.array(sunlight)
    soil_quality_arr = np.array(soil_quality)

    # 基础产量
    base_yield = 5.0

    # 温度因子：最适25℃，偏离则下降，最小为0
    temp_factor = np.maximum(0.0, 1.0 - np.abs(temp_arr - 25.0) / 25.0)

    # 降水因子：最适600mm，偏离则下降，最小为0
    rainfall_factor = np.maximum(0.0, 1.0 - np.abs(rainfall_arr - 600.0) / 600.0)

    # 施肥因子：线性正相关
    fertilizer_factor = 1.0 + fertilizer_arr / 200.0

    # 光照因子：0~12小时线性增长，上限12小时对应因子1.2
    sunlight_clipped = np.minimum(sunlight_arr, 12.0)
    sunlight_factor = 0.8 + (sunlight_clipped / 12.0) * 0.4

    # 土壤因子：直接加成
    soil_factor = 1.0 + soil_quality_arr

    # 综合预测产量
    yield_prediction = (
        base_yield *
        temp_factor *
        rainfall_factor *
        fertilizer_factor *
        sunlight_factor *
        soil_factor
    )

    # 确保产量非负
    yield_prediction = np.maximum(0.0, yield_prediction)

    return yield_prediction.tolist()
```

示例五：0b9a5088-426e-4c65-8bfb-ef98fc410da.pdf


```python
import numpy as np
from typing import List, Annotated

def predict_crop_yield_mcp(
    F: Annotated[List[float], "土壤肥力指数（建议范围：0.0 ~ 1.0，或实际评分）"],
    I: Annotated[List[float], "每周灌溉量（单位：mm/week）"],
    T: Annotated[List[float], "平均气温（单位：℃）"],
    a: Annotated[float, "土壤肥力对产量的贡献系数（默认可设为 1000）"] = 1000.0,
    b: Annotated[float, "灌溉量对产量的贡献系数（默认可设为 50）"] = 50.0,
    c: Annotated[float, "温度抑制效应系数（默认可设为 2.0）"] = 2.0
) -> Annotated[List[float], "预测单位面积作物产量（单位：kg/ha）"]:
    """
    MCP Tool：基于确定性方程预测单位面积作物产量。
    
    公式：Y = a * F + b * I - c * T²
    
    参数:
        F: 土壤肥力指数列表
        I: 每周灌溉量列表
        T: 平均气温列表
        a, b, c: 经验系数（可调参，默认值已设定）
    
    返回:
        预测产量列表（kg/ha），结果可能为负值（表示极端高温抑制严重）
    """
    # 转换为 NumPy 数组以便向量化计算
    F_arr = np.array(F)
    I_arr = np.array(I)
    T_arr = np.array(T)

    # 应用确定性模型
    yield_prediction = (
        a * F_arr +
        b * I_arr -
        c * (T_arr ** 2)
    )

    return yield_prediction.tolist()
```

示例六：0b579473-43f3-4f7d-a45a-312089b766a.pdf

```python
import numpy as np
from typing import Annotated

def calculate_evaporated_water_mcp(
    M0: Annotated[float, "初始水分含量（单位：kg 或 g）"],
    k: Annotated[float, "水分蒸发速率常数（单位：1/时间，如 1/hour）"],
    T: Annotated[float, "干燥时间（单位：小时或分钟）"]
) -> Annotated[float, "在时间 T 内累计蒸发的水分量（单位：与 M0 一致）"]:
    """
    MCP Tool：基于指数衰减模型计算食品干燥过程中累计蒸发的水分量。
    
    公式：Evaporated(T) = M0 * (T + (e^(-k*T) - 1) / k)
    
    参数:
        M0: 初始水分含量
        k: 水分蒸发速率常数
        T: 干燥总时间
    
    返回:
        累计蒸发水分量
    """
    if k == 0:
        # 避免除零，当 k=0 时，蒸发速率为0，无水分蒸发
        return 0.0
    evaporated = M0 * (T + (np.exp(-k * T) - 1) / k)
    return float(evaporated)
```
示例七：0ba15b17-85d2-4944-9a04-a9bd23c2e3f.pdf
```python
import numpy as np
from typing import Annotated

def predict_student_score_mcp(
    x1: Annotated[float, "学习时长（单位：小时）"],
    x2: Annotated[float, "出勤率（百分比，0~100）"],
    x3: Annotated[float, "平时测验平均分（百分比，0~100）"],
    x4: Annotated[float, "课堂参与度（1~5分）"],
    w1: Annotated[float, "学习时长权重系数"],
    w2: Annotated[float, "出勤率权重系数"],
    w3: Annotated[float, "测验成绩权重系数"],
    w4: Annotated[float, "课堂参与度权重系数"],
    alpha: Annotated[float, "控制S型曲线陡峭程度的参数（>0）"],
    beta: Annotated[float, "控制S型曲线在横轴上的平移位置"]
) -> Annotated[float, "预测学生得分（0~100区间）"]:
    """
    MCP Tool：基于加权Sigmoid模型预测学生课程表现得分。
    
    公式：Score = 100 / (1 + exp(-alpha * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - beta)))
    
    参数:
        x1, x2, x3, x4: 四个核心输入变量
        w1, w2, w3, w4: 对应变量的权重系数
        alpha: 控制S型曲线陡峭程度
        beta: 控制S型曲线水平偏移
    
    返回:
        预测得分（0~100），模拟学习效果的饱和与非线性提升
    """
    linear_combination = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 - beta
    score = 100.0 / (1.0 + np.exp(-alpha * linear_combination))
    return float(score)
```

示例八：0bcccdd0-b9a4-4f9b-afc2-d14b4384098.pdf

```python  
from typing import Annotated

def predict(
    a: Annotated[float, "weight coefficient for input x1"],
    b: Annotated[float, "weight coefficient for previous output y_{t-1}"],
    c: Annotated[float, "weight coefficient for second previous output y_{t-2}"],
    d: Annotated[float, "weight coefficient for product of inputs x2*x3"],
    x1: Annotated[float, "current input value x_{1,t}"],
    x2: Annotated[float, "current input value x_{2,t}"],
    x3: Annotated[float, "current input value x_{3,t}"],
    y_prev1: Annotated[float, "output value at time step t-1 (y_{t-1})"],
    y_prev2: Annotated[float, "output value at time step t-2 (y_{t-2})"],
) -> Annotated[float, "predicted output value at current time step (y_t)"]:
    return a * x1 + b * y_prev1 + c * y_prev2 + d * x2 * x3

```

示例九：00c186c3-2266-4a12-a37b-9b740fb6a97.pdf

```python
from typing import Annotated, Tuple, Callable

def calculate_DO_concentration(
    current_c: Annotated[float, "current dissolved oxygen concentration at time t"],
    prev_c: Annotated[float, "dissolved oxygen concentration at previous spatial point"],
    next_c: Annotated[float, "dissolved oxygen concentration at next spatial point"],
    D: Annotated[float, "diffusion coefficient"],
    k: Annotated[float, "gas exchange rate coefficient"],
    C_sat: Annotated[float, "temperature-related saturated dissolved oxygen concentration"],
    R_value: Annotated[float, "oxygen consumption rate function value determined by bio_load and temp"],
    dt: Annotated[float, "time step size"],
    dx: Annotated[float, "spatial step size"],
) -> Annotated[float, "next dissolved oxygen concentration at time t+dt"]:
    """
    Calculate the dissolved oxygen concentration at the next time step 
using the discretized equation.
    
    The discrete equation is derived from:
    ∂C/∂t = D * ∂²C/∂x² - k * (C - C_sat) + R(bio_load, temp)
    
    Numerically implemented using:
    - Forward difference for time (implicit or explicit Euler)
    - Central difference for space (second-order accurate)
    """
    # Compute the spatial second derivative using central difference
    d2c_dx2 = (next_c - 2 * current_c + prev_c) / (dx ** 2)
    
    # Compute the time derivative using forward difference
    dC_dt = D * d2c_dx2 - k * (current_c - C_sat) + R_value
    
    # Update concentration for the next time step
    return current_c + dt * dC_dt

```
示例十： 0d2f19ba-1875-4057-b804-379367fedec.pdf

```python
import numpy as np
from typing import Annotated, Union

def quadratic_model(x: Annotated[Union[float, np.ndarray], 'Input x. Can be a single float or a numpy array']) -> Annotated[Union[float, np.ndarray], 'Output y. Same type as input x']:
    return 2 * x**2 + 3 * x + 1
```

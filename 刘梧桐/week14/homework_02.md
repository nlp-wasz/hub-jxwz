- 解析文档：00ac792a-04dd-4639-abbd-d7f78cbb7ea.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray

def calculate_complex_system_response(
    x: Annotated[Union[float, np.ndarray], "输入变量x（无量纲，可代表如环境因子1的强度/频率等）"],
    y: Annotated[Union[float, np.ndarray], "输入变量y（无量纲，可代表如环境因子2的强度/频率等）"]
) -> Annotated[Union[float, np.ndarray], "系统目标输出值（反映x和y耦合作用的响应结果）"]:
    """
    计算复杂系统中两个输入变量x、y耦合作用下的目标输出值，融合非线性主效应与线性交互效应。
    
    公式：fun(x, y) = 2.5·sin(x) + 1.8·cos(y) + 0.3·x·y
    
    参数:
        x: 输入变量x，支持标量或多维数组（numpy.ndarray），代表系统的核心输入因子1
        y: 输入变量y，支持标量或多维数组（numpy.ndarray），代表系统的核心输入因子2
    
    返回:
        系统响应输出值，维度与输入x/y保持一致（若x/y维度不同，需满足numpy广播规则）
    
    示例:
        >>> calculate_complex_system_response(x=1.0, y=2.0)
        2.5*np.sin(1) + 1.8*np.cos(2) + 0.3*1*2 ≈ 2.12 + (-0.89) + 0.6 ≈ 1.83
        >>> calculate_complex_system_response(x=np.array([1,2]), y=np.array([3,4]))
        array([2.5*sin(1)+1.8*cos(3)+0.9, 2.5*sin(2)+1.8*cos(4)+2.4])
    """
    return 2.5 * np.sin(x) + 1.8 * np.cos(y) + 0.3 * x * y
```
- 解析文档：0bcccdd0-b9a4-4f9b-afc2-d14b4384098.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray

def calculate_difference_equation(
    x1_t: Annotated[Union[float, np.ndarray], "当前时刻t的输入变量x1（无量纲，代表外部输入因子1）"],
    x2_t: Annotated[Union[float, np.ndarray], "当前时刻t的输入变量x2（无量纲，代表外部输入因子2）"],
    x3_t: Annotated[Union[float, np.ndarray], "当前时刻t的输入变量x3（无量纲，代表外部输入因子3）"],
    y_t_1: Annotated[Union[float, np.ndarray], "前一个时间步t-1的输出值（系统历史状态1）"],
    y_t_2: Annotated[Union[float, np.ndarray], "前两个时间步t-2的输出值（系统历史状态2）"],
    a: Annotated[float, "x1_t的权重系数（调节外部输入1对当前输出的影响）"],
    b: Annotated[float, "y_t-1的权重系数（调节前1步状态对当前输出的反馈强度）"],
    c: Annotated[float, "y_t-2的权重系数（调节前2步状态对当前输出的反馈强度）"],
    d: Annotated[float, "x2_t·x3_t交互项的权重系数（调节输入2和3耦合作用的影响）"]
) -> Annotated[Union[float, np.ndarray], "当前时刻t的系统输出值y_t"]:
    """
    基于差分方程计算多输入动态系统当前时刻的输出值，融合外部输入与历史状态反馈。
    
    公式：y_t = a·x1,t + b·y_t-1 + c·y_t-2 + d·x2,t·x3,t
    
    参数:
        x1_t: 当前时刻t的输入变量x1，支持标量或numpy数组（批量计算）
        x2_t: 当前时刻t的输入变量x2，支持标量或numpy数组，与x1_t维度需匹配（满足广播规则）
        x3_t: 当前时刻t的输入变量x3，支持标量或numpy数组，与x2_t维度需匹配（满足广播规则）
        y_t_1: 前1个时间步t-1的系统输出值，反映系统近期历史状态
        y_t_2: 前2个时间步t-2的系统输出值，反映系统远期历史状态
        a: x1_t的权重系数，可正可负，代表输入1对输出的基础贡献
        b: y_t-1的权重系数，控制近期历史状态的反馈强度
        c: y_t-2的权重系数，控制远期历史状态的反馈强度
        d: x2_t·x3_t交互项的权重系数，调节两个输入变量的耦合效应
    
    返回:
        y_t: 当前时刻t的系统输出值，维度与输入参数保持一致（符合numpy广播规则）
    
    示例:
        >>> # 单样本计算
        >>> calculate_difference_equation(x1_t=2.0, x2_t=1.5, x3_t=3.0, y_t_1=4.0, y_t_2=2.5, a=0.5, b=0.8, c=0.2, d=0.1)
        0.5*2 + 0.8*4 + 0.2*2.5 + 0.1*(1.5*3) = 1 + 3.2 + 0.5 + 0.45 = 5.15
        >>> # 批量计算（数组输入）
        >>> x1 = np.array([1, 2]); x2 = np.array([3, 4]); x3 = np.array([5, 6])
        >>> y1 = np.array([7, 8]); y2 = np.array([9, 10]); a=0.1; b=0.2; c=0.3; d=0.4
        >>> calculate_difference_equation(x1, x2, x3, y1, y2, a, b, c, d)
        array([0.1*1 + 0.2*7 + 0.3*9 + 0.4*(3*5), 0.1*2 + 0.2*8 + 0.3*10 + 0.4*(4*6)])
        array([0.1+1.4+2.7+6. , 0.2+1.6+3. +9.6]) = array([10.2, 14.4])
    """
    return a * x1_t + b * y_t_1 + c * y_t_2 + d * x2_t * x3_t
```
- 解析文档：0daef473-e660-4984-be4d-940433aa889.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real

def calculate_cattle_population_growth(
    N_t: Annotated[Union[Real, np.ndarray], "第t年的牛群数量（头，非负）"],
    r: Annotated[float, "牛群年自然增长率（无量纲，通常0<r<1，代表单位数量种群的年增长比例）"],
    K: Annotated[float, "环境承载能力（头，非负，代表资源约束下的最大种群数量）"]
) -> Annotated[Union[Real, np.ndarray], "第t+1年的牛群数量（头）"]:
    """
    基于逻辑斯蒂增长模型的一阶非线性差分方程，计算牛群数量的年度动态变化。
    
    公式：N_{t+1} = N_t + r·N_t·(1 - N_t/K)
    （等价形式：N_{t+1} = N_t·[1 + r·(1 - N_t/K)]，更直观体现增长逻辑）
    
    参数:
        N_t: 第t年的牛群数量，支持标量（单种群）或numpy数组（多区域/多批次种群），需≥0
        r: 年自然增长率，反映无资源约束时的潜在增长能力，r>0为增长、r=0为稳定、r<0为衰减
        K: 环境承载能力，由饲料、场地等资源决定的最大可持续牛群数量，需>0（避免除零错误）
    
    返回:
        N_{t+1}: 第t+1年的牛群数量，维度与N_t一致，非负数值
    
    异常处理:
        若K≤0会触发值错误，若N_t<0会触发值错误，保证模型物理意义的合理性
    
    示例:
        >>> # 单种群计算：初始数量100头，增长率0.2，承载能力500头
        >>> calculate_cattle_population_growth(N_t=100, r=0.2, K=500)
        100 + 0.2*100*(1 - 100/500) = 100 + 20*0.8 = 116.0
        
        >>> # 多区域批量计算：3个区域的初始种群数量
        >>> N_t_batch = np.array([80, 200, 450])
        >>> calculate_cattle_population_growth(N_t_batch, r=0.15, K=500)
        array([80 + 0.15*80*(1-80/500), 200 + 0.15*200*(1-200/500), 450 + 0.15*450*(1-450/500)])
        array([99.2, 230. , 456.75])
    """
    # 输入合法性校验，保证模型物理意义
    if isinstance(N_t, np.ndarray):
        if np.any(N_t < 0):
            raise ValueError("牛群数量N_t不能为负数，请检查输入")
    else:
        if N_t < 0:
            raise ValueError("牛群数量N_t不能为负数，请检查输入")
    
    if K <= 0:
        raise ValueError("环境承载能力K必须大于0，否则无实际物理意义")
    
    # 核心计算公式
    return N_t + r * N_t * (1 - N_t / K)
```
- 解析文档：0f762aae-7bf5-4b3f-b38e-e9ac8eee9c9.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real

def calculate_house_market_value(
    Area: Annotated[Union[Real, np.ndarray], "房屋建筑面积（平方米，非负）"],
    Floor: Annotated[Union[Real, np.ndarray], "房屋楼层数（层，非负整数/浮点数，如3.5代表3楼半）"],
    Age: Annotated[Union[Real, np.ndarray], "房产房龄（年，非负，反映房屋折旧程度）"]
) -> Annotated[Union[Real, np.ndarray], "房产市场价值（元）"]:
    """
    基于经验模型估算房产市场价值，综合考虑面积、楼层溢价和房龄折旧的影响。
    
    公式：Value = 10000 × Area × (1 + 0.02 × Floor) × (1 − 0.015 × Age)
    公式解读：
        - 10000：基础单价（元/平方米），为经验基准值；
        - (1 + 0.02 × Floor)：楼层溢价系数，每增加1层，价值提升2%；
        - (1 − 0.015 × Age)：房龄折旧系数，每增加1年，价值折旧1.5%。
    
    参数:
        Area: 房屋建筑面积，支持标量（单套房产）或numpy数组（多套批量评估），需≥0
        Floor: 房屋楼层数，支持小数（如跃层/夹层场景），需≥0；若为负会触发值错误
        Age: 房产房龄，支持小数（如1.5年），需≥0；折旧系数最低为0（避免价值为负）
    
    返回:
        Value: 房产市场价值（元），维度与输入参数一致，非负数值
    
    异常处理:
        若Area/Floor/Age为负数，触发值错误；房龄折旧系数被限制为≥0，避免价值为负。
    
    示例:
        >>> # 单套房产评估：100㎡，10楼，5年房龄
        >>> calculate_house_market_value(Area=100, Floor=10, Age=5)
        10000 × 100 × (1+0.02×10) × (1-0.015×5) = 10000×100×1.2×0.925 = 1,110,000.0 元
        
        >>> # 多套房产批量评估
        >>> areas = np.array([80, 120, 90])
        >>> floors = np.array([5, 15, 8])
        >>> ages = np.array([3, 8, 2])
        >>> calculate_house_market_value(areas, floors, ages)
        array([10000×80×1.1×0.955, 10000×120×1.3×0.88, 10000×90×1.16×0.97])
        array([836400., 1372800., 1019880.])
    """
    # 输入合法性校验，保证模型物理意义
    if isinstance(Area, np.ndarray):
        if np.any(Area < 0):
            raise ValueError("房屋建筑面积Area不能为负数，请检查输入")
    else:
        if Area < 0:
            raise ValueError("房屋建筑面积Area不能为负数，请检查输入")
    
    if isinstance(Floor, np.ndarray):
        if np.any(Floor < 0):
            raise ValueError("楼层数Floor不能为负数，请检查输入")
    else:
        if Floor < 0:
            raise ValueError("楼层数Floor不能为负数，请检查输入")
    
    if isinstance(Age, np.ndarray):
        if np.any(Age < 0):
            raise ValueError("房龄Age不能为负数，请检查输入")
        # 限制折旧系数≥0，避免价值为负（如房龄超过66.67年时，折旧系数取0）
        depreciation_coeff = np.maximum(1 - 0.015 * Age, 0)
    else:
        if Age < 0:
            raise ValueError("房龄Age不能为负数，请检查输入")
        depreciation_coeff = max(1 - 0.015 * Age, 0)
    
    # 楼层溢价系数
    floor_premium_coeff = 1 + 0.02 * Floor
    
    # 核心计算公式
    return 10000 * Area * floor_premium_coeff * depreciation_coeff
```
- 解析文档：0fe5792f-9751-4178-b548-ddc13170a1d.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real

def calculate_food_texture_hardness(
    temp: Annotated[Union[Real, np.ndarray], "加工温度（℃，需在食品加工合理区间，如0-100℃）"],
    time: Annotated[Union[Real, np.ndarray], "加工时间（分钟，非负）"],
    ph: Annotated[Union[Real, np.ndarray], "原料pH值（无量纲，通常1-14）"],
    sugar: Annotated[Union[Real, np.ndarray], "糖分含量（g/100g，非负，代表每100克原料中的糖分量）"]
) -> Annotated[Union[Real, np.ndarray], "食品质地硬度（无量纲，数值越高硬度越大）"]:
    """
    基于线性模型预测食品加工后的质地硬度，综合考虑加工温度、时间、原料pH值和糖分含量的影响。
    
    公式：Texture Hardness = 0.5 × temp + 1.2 × time − 3.0 × ph + 0.8 × sugar
    
    参数:
        temp: 加工温度，支持标量（单批次加工）或numpy数组（多批次批量预测），需为合理加工温度
        time: 加工时间，支持标量/数组，需≥0（时间无负数值）
        ph: 原料pH值，支持标量/数组，需在1-14的合理区间（超出会提示警告）
        sugar: 糖分含量，支持标量/数组，需≥0（含量无负数值）
    
    返回:
        Texture Hardness: 食品质地硬度值，维度与输入参数一致；数值为负时代表质地偏软，正数越大硬度越高
    
    异常与警告处理:
        - 若time/sugar为负数，触发值错误；
        - 若ph超出1-14区间，触发警告（非致命），提示pH值异常；
        - 若temp为负数，触发警告，提示加工温度不合理。
    
    示例:
        >>> # 单批次加工预测：温度80℃，时间20分钟，pH6.5，糖分15g/100g
        >>> calculate_food_texture_hardness(temp=80, time=20, ph=6.5, sugar=15)
        0.5×80 + 1.2×20 - 3.0×6.5 + 0.8×15 = 40 + 24 - 19.5 + 12 = 56.5
        
        >>> # 多批次批量预测
        >>> temps = np.array([70, 85, 90])
        >>> times = np.array([15, 25, 22])
        >>> phs = np.array([5.5, 7.0, 6.0])
        >>> sugars = np.array([10, 18, 12])
        >>> calculate_food_texture_hardness(temps, times, phs, sugars)
        array([0.5×70+1.2×15-3×5.5+0.8×10, 0.5×85+1.2×25-3×7+0.8×18, 0.5×90+1.2×22-3×6+0.8×12])
        array([35+18-16.5+8, 42.5+30-21+14.4, 45+26.4-18+9.6]) = array([44.5, 65.9, 63. ])
    """
    import warnings
    
    # 输入合法性校验 - 致命错误（数值无物理意义）
    if isinstance(time, np.ndarray):
        if np.any(time < 0):
            raise ValueError("加工时间time不能为负数，请检查输入")
    else:
        if time < 0:
            raise ValueError("加工时间time不能为负数，请检查输入")
    
    if isinstance(sugar, np.ndarray):
        if np.any(sugar < 0):
            raise ValueError("糖分含量sugar不能为负数，请检查输入")
    else:
        if sugar < 0:
            raise ValueError("糖分含量sugar不能为负数，请检查输入")
    
    # 输入合理性警告 - 非致命但不符合实际加工逻辑
    if isinstance(temp, np.ndarray):
        if np.any(temp < 0):
            warnings.warn("加工温度temp出现负值，不符合食品加工常规温度区间（0-100℃），请确认输入", UserWarning)
    else:
        if temp < 0:
            warnings.warn("加工温度temp为负值，不符合食品加工常规温度区间（0-100℃），请确认输入", UserWarning)
    
    if isinstance(ph, np.ndarray):
        if np.any((ph < 1) | (ph > 14)):
            warnings.warn("部分pH值超出1-14的常规区间，可能导致硬度预测结果失真", UserWarning)
    else:
        if ph < 1 or ph > 14:
            warnings.warn("pH值超出1-14的常规区间，可能导致硬度预测结果失真", UserWarning)
    
    # 核心计算公式
    return 0.5 * temp + 1.2 * time - 3.0 * ph + 0.8 * sugar
```
- 解析文档：1b3c05f5-0953-4393-b352-ffb78279cf9.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real
import warnings

def calculate_pollutant_concentration_rate(
    C: Annotated[Union[Real, np.ndarray], "湖泊当前污染物浓度（mg/L，非负）"],
    Q_in: Annotated[Union[Real, np.ndarray], "进水流量（m³/天，非负）"],
    C_in: Annotated[Union[Real, np.ndarray], "进水污染物浓度（mg/L，非负）"],
    V: Annotated[Union[Real, np.ndarray], "湖泊总体积（m³，大于0）"],
    k: Annotated[Union[Real, np.ndarray], "污染物降解速率常数（1/天，非负）"]
) -> Annotated[Union[Real, np.ndarray], "污染物浓度随时间的变化率 dC/dt（mg/(L·天)）"]:
    """
    计算湖泊中污染物浓度的时间变化率，基于考虑进水输入和自然降解的常微分方程模型。
    
    公式：dC/dt = [Q_in × (C_in - C)] / V - k × C
    公式解读：
        - [Q_in × (C_in - C)] / V：进水输入导致的浓度变化项（输入>当前浓度时为正，反之负）；
        - -k×C：污染物自然降解导致的浓度衰减项（恒为非正）。
    
    参数:
        C: 当前污染物浓度，支持标量（单湖泊/单时间点）或numpy数组（多湖泊/批量计算），需≥0
        Q_in: 进水流量，支持标量/数组，需≥0（流量无负数值）
        C_in: 进水污染物浓度，支持标量/数组，需≥0（浓度无负数值）
        V: 湖泊总体积，支持标量/数组，需>0（体积为0无物理意义）
        k: 降解速率常数，支持标量/数组，需≥0（降解速率非负）
    
    返回:
        dC/dt: 污染物浓度变化率，正值表示浓度上升，负值表示浓度下降，维度与输入参数一致
    
    异常与警告处理:
        - 若C/Q_in/C_in/k为负数，触发值错误；
        - 若V≤0，触发值错误；
        - 若Q_in=0且C_in≠0，触发警告（无进水时进水浓度无意义）。
    
    示例:
        >>> # 单湖泊计算：当前浓度5mg/L，进水流量1000m³/天，进水浓度10mg/L，体积1e6m³，降解常数0.05/天
        >>> calculate_pollutant_concentration_rate(C=5, Q_in=1000, C_in=10, V=1e6, k=0.05)
        [1000×(10-5)]/1e6 - 0.05×5 = 5000/1e6 - 0.25 = 0.005 - 0.25 = -0.245 mg/(L·天)
        
        >>> # 多湖泊批量计算
        >>> Cs = np.array([3, 7, 4])
        >>> Q_ins = np.array([800, 1200, 900])
        >>> C_ins = np.array([8, 6, 9])
        >>> Vs = np.array([8e5, 1.2e6, 9e5])
        >>> ks = np.array([0.04, 0.06, 0.03])
        >>> calculate_pollutant_concentration_rate(Cs, Q_ins, C_ins, Vs, ks)
        array([(800×(8-3))/8e5 - 0.04×3, (1200×(6-7))/1.2e6 - 0.06×7, (900×(9-4))/9e5 - 0.03×4])
        array([0.005 - 0.12, -0.001 - 0.42, 0.005 - 0.12]) = array([-0.115, -0.421, -0.115])
    """
    # 输入合法性校验 - 致命错误（无物理意义）
    for param, name in zip([C, Q_in, C_in, k], ["C", "Q_in", "C_in", "k"]):
        if isinstance(param, np.ndarray):
            if np.any(param < 0):
                raise ValueError(f"{name}（{name}）不能为负数，请检查输入")
        else:
            if param < 0:
                raise ValueError(f"{name}（{name}）不能为负数，请检查输入")
    
    if isinstance(V, np.ndarray):
        if np.any(V <= 0):
            raise ValueError("湖泊总体积V必须大于0，体积为0无实际物理意义")
    else:
        if V <= 0:
            raise ValueError("湖泊总体积V必须大于0，体积为0无实际物理意义")
    
    # 合理性警告 - 非致命但逻辑矛盾
    if isinstance(Q_in, np.ndarray):
        zero_flow_mask = (Q_in == 0) & (C_in != 0)
        if np.any(zero_flow_mask):
            warnings.warn("存在进水流量为0但进水污染物浓度非0的情况，该参数组合无实际意义", UserWarning)
    else:
        if Q_in == 0 and C_in != 0:
            warnings.warn("进水流量为0但进水污染物浓度非0，该参数组合无实际意义", UserWarning)
    
    # 核心计算公式
    input_term = (Q_in * (C_in - C)) / V  # 进水导致的浓度变化项
    degradation_term = k * C             # 降解导致的浓度变化项
    dC_dt = input_term - degradation_term
    
    return dC_dt

```
- 解析文档：1c9ce1e1-fbc6-41bb-8e2f-83e726509e9.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real
import warnings

def calculate_environmental_quality_index(
    pollution_level: Annotated[Union[Real, np.ndarray], "污染水平（无量纲，≥0，数值越高污染越严重）"],
    population_density: Annotated[Union[Real, np.ndarray], "人口密度（人/平方公里，≥0）"],
    green_coverage: Annotated[Union[Real, np.ndarray], "绿化覆盖率（无量纲，0≤值≤1，如0.3代表30%）"]
) -> Annotated[Union[Real, np.ndarray], "环境质量指数EQI（无量纲，0~100，数值越高环境质量越好）"]:
    """
    计算环境质量指数（EQI），综合考虑污染水平、人口密度和绿化覆盖率的非线性交互作用。
    
    公式：EQI = 100 - [pollution_level / (1 + population_density)] × (1 - green_coverage)
    公式解读：
        - 100：EQI满分基准值，代表理想环境质量；
        - [pollution_level / (1 + population_density)]：人口密度调节后的污染贡献项（人口密度越高，污染对EQI的负面影响被适度稀释）；
        - (1 - green_coverage)：绿化覆盖率的调节项（绿化越高，污染对EQI的负面影响越小）；
        - 整体逻辑：污染越低、绿化越高、人口密度越小，EQI越接近100。
    
    参数:
        pollution_level: 污染水平，支持标量（单区域）或numpy数组（多区域批量评估），需≥0
        population_density: 人口密度，支持标量/数组，需≥0（人口密度无负数值）
        green_coverage: 绿化覆盖率，支持标量/数组，需在0~1区间（超出会自动截断并提示警告）
    
    返回:
        EQI: 环境质量指数，维度与输入参数一致；结果被限制在0~100区间（避免无意义的数值）
    
    异常与警告处理:
        - 若pollution_level/population_density为负数，触发值错误；
        - 若green_coverage超出0~1区间，触发警告并自动截断至0或1；
        - 计算结果若超出0~100，自动截断并提示警告（极端参数组合导致）。
    
    示例:
        >>> # 单区域评估：污染水平20，人口密度500人/平方公里，绿化覆盖率0.4（40%）
        >>> calculate_environmental_quality_index(pollution_level=20, population_density=500, green_coverage=0.4)
        100 - [20/(1+500)] × (1-0.4) = 100 - (20/501)×0.6 ≈ 100 - 0.024 = 99.976
        
        >>> # 多区域批量评估
        >>> pollutions = np.array([15, 30, 10])
        >>> densities = np.array([800, 1200, 400])
        >>> greens = np.array([0.3, 0.5, 0.6])
        >>> calculate_environmental_quality_index(pollutions, densities, greens)
        array([100 - (15/801)×0.7, 100 - (30/1201)×0.5, 100 - (10/401)×0.4])
        array([100 - 0.0131, 100 - 0.0125, 100 - 0.00998]) ≈ array([99.9869, 99.9875, 99.9900])
    """
    # 输入合法性校验 - 致命错误（无物理意义）
    for param, name in zip([pollution_level, population_density], ["pollution_level", "population_density"]):
        if isinstance(param, np.ndarray):
            if np.any(param < 0):
                raise ValueError(f"{name}不能为负数，请检查输入（{name}代表污染/人口密度，非负为物理意义）")
        else:
            if param < 0:
                raise ValueError(f"{name}不能为负数，请检查输入（{name}代表污染/人口密度，非负为物理意义）")
    
    # 绿化覆盖率合理性处理（截断+警告）
    if isinstance(green_coverage, np.ndarray):
        # 截断超出0~1的数值
        green_coverage_clipped = np.clip(green_coverage, 0, 1)
        # 检查是否有截断情况
        if np.any((green_coverage < 0) | (green_coverage > 1)):
            warnings.warn("部分绿化覆盖率超出0~1区间，已自动截断至0或1（绿化覆盖率应为0~1的比例值）", UserWarning)
    else:
        # 标量处理
        green_coverage_clipped = np.clip(green_coverage, 0, 1)
        if green_coverage < 0 or green_coverage > 1:
            warnings.warn("绿化覆盖率超出0~1区间，已自动截断至0或1（绿化覆盖率应为0~1的比例值）", UserWarning)
    
    # 核心计算公式
    pollution_adjusted = pollution_level / (1 + population_density)  # 人口密度调节后的污染项
    green_adjusted = 1 - green_coverage_clipped                     # 绿化调节项
    eqi = 100 - pollution_adjusted * green_adjusted
    
    # EQI值截断至0~100区间（避免极端参数导致无意义结果）
    eqi_clipped = np.clip(eqi, 0, 100)
    if isinstance(eqi, np.ndarray):
        if np.any((eqi < 0) | (eqi > 100)):
            warnings.warn("部分EQI计算结果超出0~100区间，已自动截断（极端参数组合导致）", UserWarning)
    else:
        if eqi < 0 or eqi > 100:
            warnings.warn("EQI计算结果超出0~100区间，已自动截断（极端参数组合导致）", UserWarning)
    
    return eqi_clipped
```
- 解析文档：1d4bb809-3292-4e24-9518-30a47c03565.pdf
```python
import numpy as np
from typing import Annotated, Union, np.ndarray
from numbers import Real
import warnings

def calculate_car_fuel_consumption(
    weight: Annotated[Union[Real, np.ndarray], "汽车重量（千克，非负，需为合理整车重量范围）"]
) -> Annotated[Union[Real, np.ndarray], "汽车百公里油耗（升/100km，非负）"]:
    """
    基于线性模型预测汽车百公里油耗，量化车辆重量对燃油消耗的影响。
    
    公式：fuel_consumption = 0.001 × weight + 5
    公式解读：
        - 5：基础油耗（升/100km），代表车辆无额外重量时的基础燃油消耗；
        - 0.001：重量油耗系数（升/(100km·千克)），每增加1千克车辆重量，油耗增加0.001升/100km。
    
    参数:
        weight: 汽车重量，支持标量（单款车型）或numpy数组（多款车型批量计算），需≥0
                常规乘用车重量范围约1000-2500千克，超出该范围会触发合理性警告
    
    返回:
        fuel_consumption: 百公里油耗（升/100km），维度与输入weight一致，非负数值
    
    异常与警告处理:
        - 若weight为负数，触发值错误（重量无物理意义）；
        - 若weight超出1000-2500千克常规范围，触发警告（提示可能的输入错误或特殊车型）；
        - 若计算出的油耗为负（极端低重量场景），自动截断为0并提示警告。
    
    示例:
        >>> # 单车型计算：重量1500千克的乘用车
        >>> calculate_car_fuel_consumption(weight=1500)
        0.001×1500 + 5 = 1.5 + 5 = 6.5 升/100km
        
        >>> # 多款车型批量计算
        >>> weights = np.array([1200, 1800, 2200])
        >>> calculate_car_fuel_consumption(weights)
        array([0.001×1200+5, 0.001×1800+5, 0.001×2200+5]) = array([6.2, 6.8, 7.2])
    """
    # 输入合法性校验 - 致命错误（重量非负）
    if isinstance(weight, np.ndarray):
        if np.any(weight < 0):
            raise ValueError("汽车重量weight不能为负数，千克为非负物理量，请检查输入")
    else:
        if weight < 0:
            raise ValueError("汽车重量weight不能为负数，千克为非负物理量，请检查输入")
    
    # 重量合理性警告（常规乘用车范围）
    normal_weight_range = (1000, 2500)
    if isinstance(weight, np.ndarray):
        out_of_range = np.any((weight < normal_weight_range[0]) | (weight > normal_weight_range[1]))
        if out_of_range:
            warnings.warn(
                f"部分汽车重量超出常规乘用车范围{normal_weight_range}千克，若为特殊车型（如重型货车/微型车）可忽略此警告",
                UserWarning
            )
    else:
        if weight < normal_weight_range[0] or weight > normal_weight_range[1]:
            warnings.warn(
                f"汽车重量{weight}千克超出常规乘用车范围{normal_weight_range}千克，若为特殊车型（如重型货车/微型车）可忽略此警告",
                UserWarning
            )
    
    # 核心计算公式
    fuel_consumption = 0.001 * weight + 5
    
    # 油耗非负处理（极端低重量场景，如weight=0时油耗为5，不会触发；仅防御性处理）
    if isinstance(fuel_consumption, np.ndarray):
        fuel_consumption_clipped = np.clip(fuel_consumption, 0, None)
        if np.any(fuel_consumption < 0):
            warnings.warn("极端低重量导致油耗计算值为负，已自动截断为0", UserWarning)
    else:
        fuel_consumption_clipped = max(fuel_consumption, 0)
        if fuel_consumption < 0:
            warnings.warn("极端低重量导致油耗计算值为负，已自动截断为0", UserWarning)
    
    return fuel_consumption_clipped
```
- 解析文档：02e55900-9b81-44f2-b1de-5210365bc87.pdf
```python
import numpy as np
import sympy as sp
from scipy import integrate
from typing import Annotated, Union, np.ndarray, Tuple
from numbers import Real
import warnings

def calculate_complex_system_output(
    x1: Annotated[Union[Real, np.ndarray], "积分上限（非负，代表累积过程的时间/空间尺度）"],
    x2: Annotated[Union[Real, np.ndarray], "二次效应输入变量（实数，无单位限制）"],
    x3: Annotated[Union[Real, np.ndarray], "周期性响应输入变量（弧度，无单位限制）"],
    x4: Annotated[Union[Real, np.ndarray], "对数变换输入变量（≥-1，避免对数无意义）"],
    x5: Annotated[Union[Real, np.ndarray], "幂律变换输入变量（≥0，避免开方无意义）"],
    a: Annotated[float, "积分项振幅系数（调节累积衰减过程的强度）"],
    b: Annotated[float, "积分项衰减系数（调节累积过程的衰减速率，b>0）"],
    c: Annotated[float, "积分项分母偏移系数（调节累积过程的基线）"],
    d: Annotated[float, "二次项系数（调节x2二次效应的强度）"],
    e: Annotated[float, "正弦项系数（调节x3周期性响应的振幅）"],
    return_symbolic: Annotated[bool, "是否返回积分项的符号表达式（默认False）"] = False
) -> Annotated[Union[Real, np.ndarray, Tuple[Union[Real, np.ndarray], sp.Expr]], "系统输出y（或y+符号表达式）"]:
    """
    计算多变量非线性复杂系统的输出值，融合累积衰减积分、二次效应、周期性响应、对数/幂律变换。
    
    公式：
    y = ∫₀^x₁ [a·e^(-t/b)] / (c + t) dt + d·x₂² + e·sin(x₃) + log(x₄ + 1) + √x₅
    各分项含义：
        - 积分项：刻画具有衰减特性的累积过程（t为积分变量）；
        - d·x₂²：输入x2的二次响应效应；
        - e·sin(x₃)：输入x3的周期性波动响应；
        - log(x₄+1)：输入x4的对数变换（+1避免log(0)）；
        - √x₅：输入x5的幂律（1/2次）变换。
    
    参数:
        x1: 积分上限，支持标量/数组，需≥0（积分区间为[0, x1]）
        x2: 二次效应输入，支持标量/数组，无严格范围限制
        x3: 周期性输入，支持标量/数组，建议输入弧度值（若为角度需提前转换）
        x4: 对数输入，支持标量/数组，需≥-1（保证x4+1>0）
        x5: 幂律输入，支持标量/数组，需≥0（保证开方有意义）
        a: 积分项振幅系数，可正可负，调节累积过程的整体强度
        b: 积分项衰减系数，需>0（衰减常数，b越大衰减越慢）
        c: 积分项分母偏移系数，避免分母为0（若c≤0需保证x1 > -c）
        d: 二次项系数，调节x2二次效应的强弱
        e: 正弦项系数，调节x3周期性响应的振幅
        return_symbolic: 若为True，返回(数值结果, 积分项符号表达式)；否则仅返回数值结果
    
    返回:
        系统输出y（标量/数组），或( y, 积分项符号表达式 )（return_symbolic=True时）
    
    异常与警告处理:
        - x1/x5<0、x4<-1、b≤0触发值错误；
        - 积分分母c+t=0（t∈[0,x1]）触发警告，积分结果可能失真；
        - 数组输入维度不匹配时触发警告（需满足numpy广播规则）。
    """
    # ========== 输入合法性校验 ==========
    # 1. 致命错误校验
    for param, name, cond in [
        (x1, "x1（积分上限）", lambda v: v < 0),
        (x5, "x5（幂律输入）", lambda v: v < 0),
        (x4, "x4（对数输入）", lambda v: v < -1),
        (b, "b（衰减系数）", lambda v: v <= 0)
    ]:
        if isinstance(param, np.ndarray):
            if np.any(cond(param)):
                raise ValueError(f"{name}存在非法值（需满足：x1≥0、x5≥0、x4≥-1、b>0），请检查输入")
        else:
            if cond(param):
                raise ValueError(f"{name}={param}非法（需满足：x1≥0、x5≥0、x4≥-1、b>0），请检查输入")
    
    # 2. 积分分母合理性警告（避免c+t=0）
    if isinstance(x1, np.ndarray):
        risky_x1 = x1[(-c <= 0) & (0 <= x1)]  # t∈[0,x1]时c+t=0的情况
        if len(risky_x1) > 0:
            warnings.warn(f"部分x1={risky_x1}导致积分区间内分母c+t=0（c={c}），积分结果可能失真", UserWarning)
    else:
        if -c >= 0 and x1 >= -c:
            warnings.warn(f"x1={x1}导致积分区间[0,{x1}]内分母c+t=0（c={c}），积分结果可能失真", UserWarning)
    
    # ========== 符号表达式定义（可选） ==========
    if return_symbolic:
        t_sym = sp.Symbol('t', real=True, nonnegative=True)
        integral_expr = sp.integrate(a * sp.exp(-t_sym/b) / (c + t_sym), (t_sym, 0, x1))
    else:
        integral_expr = None
    
    # ========== 数值计算 ==========
    # 定义积分被积函数（适配标量/数组x1）
    def integrand(t: float, a: float, b: float, c: float) -> float:
        return a * np.exp(-t / b) / (c + t)
    
    # 批量/标量积分计算
    if isinstance(x1, np.ndarray):
        integral_vals = np.zeros_like(x1, dtype=np.float64)
        for idx, x1_val in enumerate(x1):
            # 数值积分（scipy.integrate.quad适配单标量积分）
            integral_val, _ = integrate.quad(integrand, 0, x1_val, args=(a, b, c))
            integral_vals[idx] = integral_val
    else:
        integral_val, _ = integrate.quad(integrand, 0, x1, args=(a, b, c))
        integral_vals = integral_val
    
    # 非线性项计算
    quadratic_term = d * np.square(x2)
    sine_term = e * np.sin(x3)
    log_term = np.log(x4 + 1)
    power_term = np.sqrt(x5)
    
    # 总输出
    y = integral_vals + quadratic_term + sine_term + log_term + power_term
    
    # ========== 返回结果 ==========
    if return_symbolic:
        return y, integral_expr
    else:
        return y

# 扩展：符号推导辅助函数（用于理解积分项的解析性质）
def get_integral_symbolic_expression(a: float, b: float, c: float, x1: Union[float, sp.Symbol] = sp.Symbol('x1')) -> sp.Expr:
    """
    获取积分项的符号表达式（无数值计算），用于公式推导和分析。
    
    参数:
        a/b/c: 积分项系数（同主函数）
        x1: 积分上限（可传入数值或符号）
    
    返回:
        积分项的sympy符号表达式
    """
    t_sym = sp.Symbol('t', real=True, nonnegative=True)
    return sp.integrate(a * sp.exp(-t_sym/b) / (c + t_sym), (t_sym, 0, x1))
```
- 解析文档：2a40261d-07de-4bd2-9173-ba6b7bfd05b.pdf
```python
import numpy as np
from scipy.integrate import solve_ivp
from typing import Annotated, Union, np.ndarray, Tuple
from numbers import Real
import warnings

def calculate_exponential_population_growth(
    P0: Annotated[Union[Real, np.ndarray], "初始人口数量（非负，代表t=0时刻的人口规模）"],
    r: Annotated[float, "人口增长率（1/时间单位，r>0增长/ r=0稳定/ r<0衰减）"],
    t: Annotated[Union[Real, np.ndarray], "时间（与增长率r单位匹配，如r为年增长率则t单位为年，非负）"],
    method: Annotated[str, "计算方式：'analytic'（解析解，默认）/'numeric'（数值解）"] = "analytic"
) -> Annotated[Union[Real, np.ndarray], "时间t对应的人口数量P(t)（非负）"]:
    """
    计算指数人口增长模型的人口数量，支持解析解（直接公式）和数值解（ODE求解）两种方式。
    
    模型核心：
    - 微分方程：dP/dt = r·P(t)（人口增长速率与当前人口成正比）
    - 解析解：P(t) = P0·e^(r·t)（资源充足下的指数增长规律）
    
    参数:
        P0: 初始人口数量，支持标量（单种群）或numpy数组（多种群批量计算），需≥0
        r: 人口增长率，单位为“1/时间”（如年增长率则r单位为1/年）；
           r>0表示人口增长，r=0人口稳定，r<0人口衰减
        t: 目标时间点，支持标量/数组，需≥0（时间无负数值）；
           若为数组，需保证与P0维度匹配（满足numpy广播规则）
        method: 计算方式，'analytic'使用解析解（快速、精准），'numeric'使用ODE数值求解（验证微分方程）
    
    返回:
        P(t): 时间t对应的人口数量，维度与输入P0/t一致，非负数值（人口无负数量）
    
    异常与警告处理:
        - P0/t为负数、method非法触发值错误；
        - 数值解与解析解偏差超过1e-5时触发警告（验证ODE求解可靠性）；
        - 极端参数（如r过大导致P(t)溢出）触发警告并返回np.inf。
    
    示例:
        >>> # 解析解计算：初始人口1000，年增长率0.02，计算10年后人口
        >>> calculate_exponential_population_growth(P0=1000, r=0.02, t=10)
        1000 × e^(0.02×10) ≈ 1000 × 1.2214 = 1221.40（保留两位小数）
        
        >>> # 多时间点批量计算
        >>> calculate_exponential_population_growth(P0=1000, r=0.02, t=np.array([5, 10, 15]))
        array([1000×e^0.1, 1000×e^0.2, 1000×e^0.3]) ≈ array([1105.17, 1221.40, 1349.86])
        
        >>> # 数值解验证（与解析解对比）
        >>> calculate_exponential_population_growth(P0=1000, r=0.02, t=10, method='numeric')
        ≈ 1221.40（与解析解偏差<1e-5）
    """
    # ========== 输入合法性校验 ==========
    # 1. 非负校验
    for param, name in zip([P0, t], ["初始人口P0", "时间t"]):
        if isinstance(param, np.ndarray):
            if np.any(param < 0):
                raise ValueError(f"{name}不能为负数（人口/时间无负数值），请检查输入")
        else:
            if param < 0:
                raise ValueError(f"{name}={param}为负数（人口/时间无负数值），请检查输入")
    
    # 2. 计算方式校验
    if method not in ['analytic', 'numeric']:
        raise ValueError(f"method={method}非法，仅支持'analytic'（解析解）或'numeric'（数值解）")
    
    # ========== 核心计算逻辑 ==========
    if method == 'analytic':
        # 解析解：P(t) = P0·e^(r·t)
        try:
            P_t = P0 * np.exp(r * t)
        except FloatingPointError:
            warnings.warn("极端参数导致人口数量计算溢出，返回无穷大", UserWarning)
            P_t = np.inf if isinstance(P0, Real) else np.full_like(P0, np.inf)
    
    else:  # numeric：数值求解微分方程 dP/dt = r·P
        # 定义ODE右侧函数（scipy.solve_ivp要求格式：dPdt(t, P)）
        def dPdt(t_val, P_val):
            return r * P_val
        
        # 适配标量/数组t的数值求解
        if isinstance(t, np.ndarray):
            P_t = np.zeros_like(t, dtype=np.float64)
            for idx, t_val in enumerate(t):
                # 求解ODE：t_span=[0, t_val], 初始值y0=[P0]
                sol = solve_ivp(
                    fun=dPdt,
                    t_span=(0, t_val),
                    y0=[P0] if isinstance(P0, Real) else [P0[idx]],
                    t_eval=[t_val],
                    method="RK45"  # 经典四阶龙格-库塔法，适配指数增长ODE
                )
                if not sol.success:
                    warnings.warn(f"时间{t_val}的ODE求解失败：{sol.message}", UserWarning)
                    P_t[idx] = np.nan
                else:
                    P_t[idx] = sol.y[0][0]
        else:
            # 标量t求解
            sol = solve_ivp(
                fun=dPdt,
                t_span=(0, t),
                y0=[P0],
                t_eval=[t],
                method="RK45"
            )
            if not sol.success:
                warnings.warn(f"ODE求解失败：{sol.message}", UserWarning)
                P_t = np.nan
            else:
                P_t = sol.y[0][0]
        
        # 数值解与解析解偏差校验（验证可靠性）
        P_analytic = P0 * np.exp(r * t)
        if isinstance(P_t, np.ndarray):
            deviation = np.abs(P_t - P_analytic) / (P_analytic + 1e-8)  # 避免除零
            if np.any(deviation > 1e-5):
                warnings.warn("数值解与解析解偏差超过1e-5，可能存在求解精度问题", UserWarning)
        else:
            deviation = abs(P_t - P_analytic) / (P_analytic + 1e-8)
            if deviation > 1e-5:
                warnings.warn("数值解与解析解偏差超过1e-5，可能存在求解精度问题", UserWarning)
    
    # ========== 人口数量非负处理 ==========
    if isinstance(P_t, np.ndarray):
        P_t_clipped = np.clip(P_t, 0, None)
        if np.any(P_t < 0):
            warnings.warn("极端参数导致人口计算值为负，已自动截断为0", UserWarning)
    else:
        P_t_clipped = max(P_t, 0)
        if P_t < 0:
            warnings.warn("极端参数导致人口计算值为负，已自动截断为0", UserWarning)
    
    return P_t_clipped

# 扩展函数：计算人口翻倍时间（仅适用于r>0的增长场景）
def calculate_population_doubling_time(
    r: Annotated[float, "人口增长率（r>0，1/时间单位）"]
) -> Annotated[float, "人口翻倍所需时间（与r单位匹配）"]:
    """
    基于指数增长模型计算人口翻倍时间（规则：t_double = ln(2)/r）。
    
    参数:
        r: 人口增长率，需>0（仅增长场景有翻倍时间）
    
    返回:
        t_double: 人口翻倍时间（如r为年增长率，则返回值单位为年）
    
    示例:
        >>> calculate_population_doubling_time(r=0.02)  # 年增长率2%
        ln(2)/0.02 ≈ 34.66 年
    """
    if r <= 0:
        raise ValueError("仅当增长率r>0时存在翻倍时间，当前r={}".format(r))
    return np.log(2) / r
```

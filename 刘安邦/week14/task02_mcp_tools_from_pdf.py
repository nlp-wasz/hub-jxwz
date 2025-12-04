"""
提示词：
你是一个文档解析与建模计算专家，请你解析这10个pdf文档，并根据解析得到的公式，生成对应的mcp可执行tool，可以使用numpy或sympy。
只需输出10个mcp tool对应python代码。
"""
import numpy as np
import sympy as sp
from typing import Union, List, Tuple


def tool1(t: Union[float, np.ndarray], k: float, u0: float) -> Union[float, np.ndarray]:
    """
    文档1: 用户数量增长模型
    描述用户数量随时间演化的指数增长趋势
    """
    return u0 * np.exp(k * t)


def tool2(amenity_quality: float, crowd_level: float, travel_time: float,
          temperature: float, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    文档2: 游客满意度变化模型
    计算游客满意度随时间的变化率
    """
    dS_dt = (0.3 * (amenity_quality / 5) -
             0.2 * (crowd_level / 10) -
             0.1 * travel_time -
             0.05 * np.abs(temperature - 22))
    return dS_dt * t  # 返回在时间t内的满意度变化量


def tool3(t: Union[float, np.ndarray], k: float, u0: float) -> Union[float, np.ndarray]:
    """
    文档3: 游客数量衰减模型
    描述游客数量随时间呈指数衰减趋势
    """
    return u0 * np.exp(-k * t)


def tool4(load: float, span: float, stiffness: float) -> float:
    """
    文档4: 结构挠度估算模型
    计算简支梁在均布荷载下的最大挠度
    """
    C = 48
    return (load * span ** 3) / (C * stiffness)


def tool5(temp: float, humidity: float) -> float:
    """
    文档5: 旅游舒适度指数模型
    计算基于气温和湿度的旅游舒适度指数
    """
    return 100 - 0.5 * (temp - 22) ** 2 - 0.2 * humidity ** 2


def tool6(price: float, discount: float, time: float,
          stock: float, rating: float) -> float:
    """
    文档6: 电商转化率预测模型
    综合多因素预测用户购买转化率
    """
    return (0.5 * np.exp(-0.1 * price) +
            0.3 * (1 - np.exp(-0.5 * discount)) +
            0.1 * np.log(1 + time) +
            0.05 * np.sqrt(stock) +
            0.05 * rating)


def tool7(length: float, width: float, height: float) -> float:
    """
    文档7: 混凝土用量估算模型
    考虑施工损耗的混凝土体积计算
    """
    return length * width * height * 1.02


def tool8(speed: float, acceleration: float, engine_load: float,
          temperature: float, road_gradient: float) -> float:
    """
    文档8: 车辆油耗预测模型
    非线性回归预测百公里油耗
    """
    return (5 + 0.02 * speed ** 2 - 0.3 * speed +
            1.5 * acceleration ** 2 + 0.01 * engine_load -
            0.005 * temperature + 0.2 * road_gradient ** 2)


def tool9(t: Union[float, np.ndarray], C0: float, k: float,
          Ea: float, T: float) -> Union[float, np.ndarray]:
    """
    文档9: 食品成分降解动力学模型
    基于Arrhenius方程的浓度变化预测
    """
    R = 8.314  # 理想气体常数
    k_temp = k * np.exp(-Ea / (R * T))
    return C0 * np.exp(-k_temp * t)


def tool10(x: np.ndarray, t: float, u0: np.ndarray, D: float,
           r: float, K: float, dx: float, dt: float) -> np.ndarray:
    """
    文档10: 种群动态演化模型（反应扩散方程）
    使用有限差分法求解一维空间中的种群密度演化
    """
    # 初始化种群密度数组
    u = u0.copy()
    n = len(x)

    # 时间步数
    nt = int(t / dt)

    for _ in range(nt):
        u_new = u.copy()

        # 使用显式有限差分法
        for i in range(1, n - 1):
            # 扩散项：D * d²u/dx²
            diffusion = D * (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
            # 反应项：r * u * (1 - u/K)
            reaction = r * u[i] * (1 - u[i] / K)
            # 更新种群密度
            u_new[i] = u[i] + dt * (diffusion + reaction)

        # 边界条件：零梯度边界
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        u = u_new

    return u
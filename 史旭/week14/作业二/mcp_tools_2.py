# 将解析的pdf文档内容，转换为可计算的 mcp 工具
import math
from fastmcp import FastMCP
from typing import Union, List

mcp = FastMCP(name="MCP 工具1")


# 工具 6
@mcp.tool()
def predict_student_score(
        study_hours: float,
        attendance_rate: float,
        quiz_average: float,
        participation: float
) -> float:
    """
    基于 Sigmoid 模型预测学生课程最终得分（0~100 分）。

    公式：
        Score = 100 / (1 + exp(-α * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - β)))

    默认参数（可校准）：
        w1=0.5, w2=0.8, w3=1.0, w4=5.0
        α=0.05, β=60

    参数:
        study_hours (float): 学习时长（小时），例如 20.0
        attendance_rate (float): 出勤率（0~100），例如 90.0
        quiz_average (float): 平时测验平均分（0~100），例如 85.0
        participation (float): 课堂参与度（1~5），例如 4.0

    返回:
        float: 预测成绩（0.0 ~ 100.0）
    """
    # 输入校验
    if not (0 <= attendance_rate <= 100):
        raise ValueError("出勤率应在 0~100 之间")
    if not (0 <= quiz_average <= 100):
        raise ValueError("测验平均分应在 0~100 之间")
    if not (1 <= participation <= 5):
        raise ValueError("课堂参与度应在 1~5 之间")
    if study_hours < 0:
        raise ValueError("学习时长不能为负")

    # 权重与超参数（可配置）
    w1, w2, w3, w4 = 0.5, 0.8, 1.0, 5.0
    alpha = 0.05
    beta = 60.0

    # 加权线性组合
    linear_comb = (
            w1 * study_hours +
            w2 * attendance_rate +
            w3 * quiz_average +
            w4 * participation
    )

    # Sigmoid 映射到 [0, 100]
    exponent = -alpha * (linear_comb - beta)
    # 防止 overflow
    if exponent > 700:
        score = 0.0
    elif exponent < -700:
        score = 100.0
    else:
        score = 100.0 / (1.0 + math.exp(exponent))

    return round(score, 2)


# 工具 7
@mcp.tool()
def predict_system_state(
        x1_t: float,
        x2_t: float,
        x3_t: float,
        y_t_minus_1: float,
        y_t_minus_2: float,
        a: float = 0.5,
        b: float = 0.6,
        c: float = 0.2,
        d: float = 0.1
) -> float:
    """
    使用二阶差分方程预测系统当前状态 y_t。

    公式：
        y_t = a*x1_t + b*y_{t-1} + c*y_{t-2} + d*(x2_t * x3_t)

    参数:
        x1_t (float): 当前时刻输入变量 1
        x2_t (float): 当前时刻输入变量 2
        x3_t (float): 当前时刻输入变量 3
        y_t_minus_1 (float): 上一时刻系统输出 y_{t-1}
        y_t_minus_2 (float): 上上时刻系统输出 y_{t-2}
        a, b, c, d (float): 模型参数（可选，默认已设）

    返回:
        float: 当前时刻预测输出 y_t
    """
    y_t = (
            a * x1_t +
            b * y_t_minus_1 +
            c * y_t_minus_2 +
            d * (x2_t * x3_t)
    )
    return y_t


# 工具 8
@mcp.tool()
def quadratic_model(x: float) -> float:
    """
    确定性二次函数模型：y = 2*x^2 + 3*x + 1

    该模型用于演示输入变量 x 与输出 y 之间的非线性映射关系，
    适用于工程计算、教学示例或系统行为模拟等场景。

    参数:
        x (float): 输入变量（任意实数）

    返回:
        float: 输出结果 y
    """
    return 2.0 * x * x + 3.0 * x + 1.0


# 工具 9
@mcp.tool()
def nonlinear_interaction_model(x: float, y: float) -> float:
    """
    计算双变量非线性交互模型的输出：
        f(x, y) = 2.5 * sin(x) + 1.8 * cos(y) + 0.3 * x * y

    该模型融合了：
      - x 的周期性主效应（正弦）
      - y 的周期性主效应（余弦）
      - x 与 y 的线性交互项（乘积）

    输入单位：x 和 y 均为**弧度**（radians）。

    参数:
        x (float): 第一个输入变量（弧度）
        y (float): 第二个输入变量（弧度）

    返回:
        float: 模型输出值（无量纲或根据上下文解释）
    """
    return (
            2.5 * math.sin(x) +
            1.8 * math.cos(y) +
            0.3 * x * y
    )


# 工具 10
@mcp.tool()
def predict_dissolved_oxygen(
        temperature: float,
        bio_load: float,
        gas_exchange_coeff: float = 0.1,
        alpha: float = 0.02
) -> dict:
    """
    预测水产养殖水体中溶解氧浓度（mg/L）的稳态近似值。

    模型基于气体交换与生物耗氧的平衡：
        C = C_sat - R / k

    其中：
      - C_sat: 温度相关的饱和溶解氧（mg/L）
      - R: 生物耗氧速率 = alpha * bio_load * temperature
      - k: 气体交换系数（1/h）

    饱和 DO 经验公式（APHA 近似，0~30℃）：
        C_sat = 14.652 - 0.41022*T + 0.007991*T^2 - 0.000077774*T^3

    参数:
        temperature (float): 水温（℃），建议范围 5~35
        bio_load (float): 生物负载（如 kg 鱼/m³ 或相对指数 ≥0）
        gas_exchange_coeff (float): 气体交换系数 k（1/h，默认 0.1）
        alpha (float): 耗氧比例系数（默认 0.02）

    返回:
        dict: {
            "dissolved_oxygen": float,   # 预测 DO 浓度（mg/L）
            "saturation_do": float,      # 饱和 DO（mg/L）
            "oxygen_consumption_rate": float  # R（mg/L/h）
        }
    """
    T = temperature
    if T < 0 or T > 40:
        raise ValueError("水温应在 0~40℃ 范围内")
    if bio_load < 0:
        raise ValueError("生物负载不能为负")
    if gas_exchange_coeff <= 0:
        raise ValueError("气体交换系数必须 > 0")

    # 1. 计算饱和溶解氧 C_sat (mg/L)
    C_sat = (
            14.652
            - 0.41022 * T
            + 0.007991 * T ** 2
            - 0.000077774 * T ** 3
    )

    # 2. 计算耗氧速率 R (mg/L/h)
    R = alpha * bio_load * T  # 简化模型

    # 3. 稳态 DO 浓度
    C = C_sat - (R / gas_exchange_coeff)

    # 4. 物理约束：DO >= 0
    C = max(0.0, C)

    return {
        "dissolved_oxygen": round(C, 3),
        "saturation_do": round(C_sat, 3),
        "oxygen_consumption_rate": round(R, 3)
    }

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Aquaculture Dissolved Oxygen Model")

@mcp.tool()
def calculate_do_concentration(
    initial_concentration: float = 8.0,        # 初始溶解氧浓度 (mg/L)
    temperature: float = 25.0,                # 水温 (°C)
    bio_load: float = 0.1,                    # 生物负载 (kg/m³)
    gas_exchange_rate: float = 0.5,           # 气体交换速率系数 k (1/day)
    diffusion_coefficient: float = 0.01,      # 扩散系数 D (m²/day)
    system_length: float = 10.0,              # 系统长度 (m)
    simulation_time: float = 7.0,             # 模拟时间 (天)
    num_points_x: int = 50,                   # 空间网格点数
    num_points_t: int = 100                   # 时间步数
) -> dict:
    """
    水平养殖系统溶解氧浓度动态预测模型
    
    建模背景:
    在水平养殖系统中，溶解氧（DO）是影响水生生物健康和生长的关键水质参数之一。
    为了实现对水体中溶解氧浓度的动态预测和调控，有必要建立能够反映其随时间和空间变化的数学模型。
    本模型基于扩散-反应机制，结合环境因素和生物活动的影响，构建了一个简化的偏微分方程框架，
    用于模拟溶解氧在养殖水体中的分布特征。该模型综合考虑了物理扩散、气体交换、
    温度对溶解氧饱和度的影响，以及生物代谢引起的氧消耗过程，适用于对水平养殖系统中的
    溶解氧变化趋势进行预测与分析。
    
    建模公式:
    模型的基本形式为一个一维瞬态偏微分方程：
    
    \[\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2} - k(C - C_{sat}) + R(bio\_load, temp)\]
    
    其中:
    - $C$: 溶解氧浓度 (mg/L)
    - $D$: 扩散系数 (m²/day)
    - $k$: 气体交换速率系数 (1/day)
    - $C_{sat}$: 温度相关的溶解氧饱和浓度 (mg/L)
    - $R$: 由生物负载和水温决定的耗氧速率函数 (mg/L/day)
    
    模型进一步通过经验关系对空间和时间效应进行参数化处理，从而构建出适用于实际应用的数值模拟函数。
    
    参数说明:
    - initial_concentration: 初始溶解氧浓度，单位：mg/L
    - temperature: 水温，单位：°C
    - bio_load: 生物负载，表示单位水体的生物量，单位：kg/m³
    - gas_exchange_rate: 气体交换速率系数，表示氧气从空气进入水体的速率，单位：1/day
    - diffusion_coefficient: 扩散系数，表示氧气在水中的扩散能力，单位：m²/day
    - system_length: 养殖系统长度，单位：m
    - simulation_time: 模拟时间长度，单位：天
    - num_points_x: 空间网格点数
    - num_points_t: 时间步数
    
    返回:
    - dict: 包含空间网格、时间序列和溶解氧浓度分布矩阵的字典
    
    使用示例:
    >>> result = calculate_do_concentration(
    ...     initial_concentration=8.0,
    ...     temperature=25.0,
    ...     bio_load=0.1,
    ...     simulation_time=7.0
    ... )
    >>> print(f"最终平均浓度: {np.mean(result['concentration'][-1, :]):.3f} mg/L")
    """
    
    # 1. 计算温度相关的溶解氧饱和浓度 (使用APHA标准经验公式)
    def calculate_saturation_do(temp):
        """计算给定温度下的溶解氧饱和浓度 (mg/L)"""
        # APHA标准经验公式
        C_sat = 14.619 - 0.4042 * temp + 0.0084 * temp**2 - 0.00009 * temp**3
        return max(C_sat, 0)
    
    # 2. 计算生物耗氧速率函数
    def calculate_respiration_rate(bio_load, temp):
        """计算由生物负载和水温决定的耗氧速率"""
        # 基础耗氧速率：与生物负载成正比
        base_rate = 0.05 * bio_load  # mg/L/day per kg/m³
        
        # 温度修正因子：温度每升高10°C，耗氧速率增加约2倍
        Q10 = 2.0
        temp_factor = Q10 ** ((temp - 20.0) / 10.0)
        
        # 总耗氧速率 (负值表示消耗氧气)
        R = -base_rate * temp_factor
        return R
    
    # 3. 创建空间网格和时间网格
    x = np.linspace(0, system_length, num_points_x)
    t = np.linspace(0, simulation_time, num_points_t)
    dx = x[1] - x[0]
    
    # 4. 计算模型参数
    C_sat = calculate_saturation_do(temperature)
    R = calculate_respiration_rate(bio_load, temperature)
    
    # 5. 使用有限差分法求解PDE
    C = np.zeros((num_points_t, num_points_x))
    C[0, :] = initial_concentration  # 初始条件
    
    # 边界条件：两端为Neumann边界条件（零梯度）
    for i in range(1, num_points_t):
        dt = t[i] - t[i-1]
        
        # 使用隐式差分法求解扩散项
        # 构建系数矩阵
        r = diffusion_coefficient * dt / (dx**2)
        main_diag = np.ones(num_points_x) * (1 + 2*r + gas_exchange_rate * dt)
        off_diag = np.ones(num_points_x - 1) * (-r)
        
        # 应用边界条件（零梯度）
        main_diag[0] = 1 + r + gas_exchange_rate * dt
        main_diag[-1] = 1 + r + gas_exchange_rate * dt
        
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
        
        # 构建右侧向量
        b = C[i-1, :] + dt * (gas_exchange_rate * C_sat + R)
        
        # 求解线性系统
        C[i, :] = spsolve(A, b)
    
    # 6. 计算统计信息
    avg_concentration = np.mean(C, axis=1)
    min_concentration = np.min(C, axis=1)
    max_concentration = np.max(C, axis=1)
    
    # 计算空间梯度（用于分析混合程度）
    spatial_gradient = np.gradient(C[-1, :], dx)
    
    # 计算系统状态评估
    critical_threshold = 5.0  # 溶解氧临界阈值 (mg/L)
    min_overall = np.min(C)
    time_below_threshold = np.sum(np.any(C < critical_threshold, axis=1)) / num_points_t * simulation_time
    
    system_status = "健康" if min_overall > critical_threshold else "需关注"
    
    return {
        "spatial_grid": x.tolist(),
        "time_grid": t.tolist(),
        "concentration": C.tolist(),
        "saturation_concentration": float(C_sat),
        "respiration_rate": float(R),
        "statistics": {
            "average_concentration_over_time": avg_concentration.tolist(),
            "minimum_concentration_over_time": min_concentration.tolist(),
            "maximum_concentration_over_time": max_concentration.tolist(),
            "final_spatial_distribution": C[-1, :].tolist(),
            "spatial_gradient": spatial_gradient.tolist()
        },
        "system_assessment": {
            "minimum_do_observed": float(min_overall),
            "time_below_critical_threshold": float(time_below_threshold),
            "critical_threshold": critical_threshold,
            "system_status": system_status,
            "recommendation": "增氧" if system_status == "需关注" else "维持"
        },
        "model_parameters": {
            "initial_concentration": initial_concentration,
            "temperature": temperature,
            "bio_load": bio_load,
            "gas_exchange_rate": gas_exchange_rate,
            "diffusion_coefficient": diffusion_coefficient,
            "system_length": system_length,
            "simulation_time": simulation_time
        }
    }

@mcp.tool()
def calculate_do_saturation(
    temperature: float = 25.0  # 水温 (°C)
) -> float:
    """
    计算给定温度下的溶解氧饱和浓度
    
    使用APHA标准经验公式:
    C_sat = 14.619 - 0.4042*T + 0.0084*T² - 0.00009*T³
    
    参数:
    - temperature: 水温，单位：°C
    
    返回:
    - 溶解氧饱和浓度，单位：mg/L
    """
    C_sat = 14.619 - 0.4042 * temperature + 0.0084 * temperature**2 - 0.00009 * temperature**3
    return max(C_sat, 0)

@mcp.tool()
def analyze_aeration_requirement(
    current_do: float = 6.0,      # 当前溶解氧浓度 (mg/L)
    target_do: float = 7.0,       # 目标溶解氧浓度 (mg/L)
    volume: float = 100.0,        # 水体体积 (m³)
    temperature: float = 25.0,    # 水温 (°C)
    bio_load: float = 0.1         # 生物负载 (kg/m³)
) -> dict:
    """
    分析增氧需求
    
    参数:
    - current_do: 当前溶解氧浓度 (mg/L)
    - target_do: 目标溶解氧浓度 (mg/L)
    - volume: 水体体积 (m³)
    - temperature: 水温 (°C)
    - bio_load: 生物负载 (kg/m³)
    
    返回:
    - dict: 增氧需求分析结果
    """
    # 计算饱和浓度
    C_sat = calculate_do_saturation(temperature)
    
    # 计算耗氧速率
    def calculate_respiration_rate(bio_load, temp):
        base_rate = 0.05 * bio_load
        Q10 = 2.0
        temp_factor = Q10 ** ((temp - 20.0) / 10.0)
        return -base_rate * temp_factor
    
    R = calculate_respiration_rate(bio_load, temperature)
    
    # 计算需增加的氧气量
    do_deficit = target_do - current_do
    oxygen_mass_needed = do_deficit * volume  # mg/L * m³ = g
    
    # 假设增氧效率为50%
    aeration_efficiency = 0.5
    actual_oxygen_needed = oxygen_mass_needed / aeration_efficiency
    
    # 考虑生物耗氧
    daily_consumption = abs(R) * volume  # 每日耗氧量 (g/day)
    
    # 计算建议增氧时间
    # 假设增氧设备功率为100 g/hour
    aeration_rate = 100.0  # g/hour
    aeration_time_hours = actual_oxygen_needed / aeration_rate
    
    return {
        "saturation_concentration": float(C_sat),
        "current_oxygen_saturation": float(current_do / C_sat * 100),
        "oxygen_deficit": float(do_deficit),
        "oxygen_mass_needed": float(oxygen_mass_needed / 1000),  # 转换为kg
        "daily_oxygen_consumption": float(daily_consumption / 1000),  # 转换为kg/day
        "recommended_aeration_time": float(aeration_time_hours),
        "aeration_rate_required": float(actual_oxygen_needed / aeration_time_hours) if aeration_time_hours > 0 else 0,
        "suggestions": [
            f"需要增加 {do_deficit:.2f} mg/L 的溶解氧",
            f"相当于 {oxygen_mass_needed/1000:.2f} kg 氧气",
            f"建议增氧时间: {aeration_time_hours:.1f} 小时",
            f"每日生物耗氧量: {daily_consumption/1000:.2f} kg"
        ]
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
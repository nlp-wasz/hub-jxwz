import numpy as np
from scipy.integrate import solve_ivp
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Agricultural Product Inventory Model")

@mcp.tool()
def simulate_inventory_dynamics(
    initial_inventory: float = 1000.0,      # 初始库存量 (kg)
    depletion_rate: float = 0.1,           # 库存表率 k (1/day)
    replenishment_strategy: str = "constant",  # 补货策略: constant, periodic, or none
    constant_replenishment_rate: float = 100.0,  # 常数补货速率 (kg/day)
    periodic_amplitude: float = 50.0,      # 周期性补货振幅 (kg/day)
    periodic_frequency: float = 1.0,       # 周期性补货频率 (1/day)
    simulation_time: float = 10.0,         # 模拟时间 (天)
    num_time_points: int = 100             # 时间点数
) -> dict:
    """
    农产品库存动态变化模型
    
    建模背景:
    在农产品批发与零售过程中，库存管理是保障供应链稳定运行的关键环节。
    由于农产品具有易腐性、季节性和供需波动性等特点，合理预测库存变化对于
    控制损耗、优化补货策略和提升运营效率具有重要意义。为此，构建一个描述
    库存随时间变化的动态模型，有助于理解库存消耗与补货之间的平衡关系，
    并为实际运营提供量化分析依据。
    
    本模型聚焦于单一农产品的库存变化过程，假设其库存量随时间自然减少，
    减少速率与当前库存水平成正比，体现了销售过程中的连续消耗特性。
    同时，考虑外部补货行为对库存的动态补充作用，将补货量作为外部输入变量
    引入模型中，从而形成一个反映库存变化的动态系统。
    
    建模公式:
    库存量 S(t) 的变化过程用一阶常微分方程描述：
    
    \[\frac{dS}{dt} = -k \cdot S(t) + u(t)\]
    
    其中：
    - S(t) 表示在时间 t 时的库存量；
    - k 为库存表率，反映销售速率；
    - u(t) 表示在时间 t 时的外部补货速率，作为系统的输入变量。
    
    该模型可用于模拟在不同补货策略下库存的动态变化情况，为进一步的库存
    优化与控制策略设计提供基础。
    
    参数说明:
    - initial_inventory: 初始库存量，单位：kg
    - depletion_rate: 库存表率 k，表示库存的日消耗率，单位：1/天
    - replenishment_strategy: 补货策略，可选值：
        * "constant": 常数补货策略
        * "periodic": 周期性补货策略
        * "none": 无补货
    - constant_replenishment_rate: 常数补货速率，单位：kg/天
    - periodic_amplitude: 周期性补货振幅，单位：kg/天
    - periodic_frequency: 周期性补货频率，单位：1/天
    - simulation_time: 模拟时间长度，单位：天
    - num_time_points: 时间点数
    
    返回:
    - dict: 包含时间序列、库存序列、关键指标和模型参数的字典
    
    使用示例:
    >>> result = simulate_inventory_dynamics(
    ...     initial_inventory=1000.0,
    ...     depletion_rate=0.1,
    ...     replenishment_strategy="constant",
    ...     constant_replenishment_rate=100.0,
    ...     simulation_time=10.0
    ... )
    >>> print(f"最终库存: {result['final_inventory']:.2f} kg")
    """
    
    # 1. 定义补货速率函数 u(t)
    def u(t):
        if replenishment_strategy == "constant":
            return constant_replenishment_rate
        elif replenishment_strategy == "periodic":
            # 正弦波形式的周期性补货
            return periodic_amplitude * (1 + np.sin(2 * np.pi * periodic_frequency * t)) / 2
        elif replenishment_strategy == "none":
            return 0.0
        else:
            raise ValueError(f"未知的补货策略: {replenishment_strategy}")
    
    # 2. 定义微分方程：dS/dt = -k*S + u(t)
    def inventory_ode(t, S):
        return -depletion_rate * S + u(t)
    
    # 3. 设置时间网格
    t_eval = np.linspace(0, simulation_time, num_time_points)
    
    # 4. 求解微分方程
    sol = solve_ivp(
        inventory_ode,
        [0, simulation_time],
        [initial_inventory],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # 5. 提取结果
    time_points = sol.t
    inventory_levels = sol.y[0]
    
    # 6. 计算关键指标
    min_inventory = np.min(inventory_levels)
    max_inventory = np.max(inventory_levels)
    avg_inventory = np.mean(inventory_levels)
    final_inventory = inventory_levels[-1]
    
    # 计算库存变化率
    inventory_changes = np.diff(inventory_levels)
    
    # 找出库存最低点的时间
    min_inventory_index = np.argmin(inventory_levels)
    time_of_min_inventory = time_points[min_inventory_index]
    
    # 计算库存耗尽的风险（库存低于安全水平的时间比例）
    safety_threshold = 0.2 * initial_inventory  # 安全库存设为初始库存的20%
    low_inventory_mask = inventory_levels < safety_threshold
    low_inventory_ratio = np.sum(low_inventory_mask) / len(inventory_levels)
    
    # 7. 计算理论稳定状态（仅对常数补货有效）
    if replenishment_strategy == "constant" and depletion_rate > 0:
        steady_state = constant_replenishment_rate / depletion_rate
        time_to_steady_state = -np.log(0.01) / depletion_rate  # 达到99%稳定状态的时间
    else:
        steady_state = None
        time_to_steady_state = None
    
    # 8. 评估库存管理效果
    if final_inventory < safety_threshold:
        inventory_status = "需补货"
    elif low_inventory_ratio > 0.3:
        inventory_status = "需关注"
    else:
        inventory_status = "良好"
    
    return {
        "time_series": time_points.tolist(),
        "inventory_series": inventory_levels.tolist(),
        "inventory_changes": inventory_changes.tolist(),
        "key_metrics": {
            "initial_inventory": float(initial_inventory),
            "final_inventory": float(final_inventory),
            "minimum_inventory": float(min_inventory),
            "maximum_inventory": float(max_inventory),
            "average_inventory": float(avg_inventory),
            "time_of_minimum_inventory": float(time_of_min_inventory),
            "low_inventory_ratio": float(low_inventory_ratio),
            "safety_threshold": float(safety_threshold),
            "inventory_status": inventory_status
        },
        "model_parameters": {
            "depletion_rate": float(depletion_rate),
            "replenishment_strategy": replenishment_strategy,
            "constant_replenishment_rate": float(constant_replenishment_rate) if replenishment_strategy == "constant" else None,
            "periodic_amplitude": float(periodic_amplitude) if replenishment_strategy == "periodic" else None,
            "periodic_frequency": float(periodic_frequency) if replenishment_strategy == "periodic" else None,
            "simulation_time": float(simulation_time)
        },
        "steady_state_analysis": {
            "steady_state_inventory": float(steady_state) if steady_state is not None else None,
            "time_to_steady_state": float(time_to_steady_state) if time_to_steady_state is not None else None,
            "is_stable": steady_state is not None and abs(final_inventory - steady_state) < 0.01 * steady_state
        },
        "management_recommendations": [
            f"库存状态: {inventory_status}",
            f"安全库存阈值: {safety_threshold:.1f} kg",
            f"库存低于安全水平的时间比例: {low_inventory_ratio*100:.1f}%",
            f"建议下次补货时间: {time_of_min_inventory:.1f} 天" if inventory_status in ["需补货", "需关注"] else "当前库存充足"
        ]
    }

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
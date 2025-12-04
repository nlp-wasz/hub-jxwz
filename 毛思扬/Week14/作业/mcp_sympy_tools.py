import sympy as sp
from fastmcp import FastMCP

# 创建MCP服务器实例
mcp = FastMCP(
    name="Sympy-Formula-Calculator-MCP",
    instructions="This server contains tools for calculating mathematical formulas using sympy."
)

@mcp.tool
def calculate_cow_milk_production_symbolic(
    feed_quality: float,
    temperature: float,
    health_score: float,
    drinking_time: float
) -> float:
    """
    根据奶牛饲养条件计算日均产奶量（使用sympy符号计算）
    
    公式: 产奶量 = 20 + 1.5 × 饲料质量 - 0.3 × (气温 - 20) + 0.05 × 健康评分 + 0.4 × 饮水时间 + ε
    其中 ε 是服从 N(0,1) 的随机扰动项，这里取期望值0
    """
    # 定义符号变量
    f, t, h, d = sp.symbols('f t h d')
    
    # 定义公式
    milk_yield_expr = 20 + 1.5 * f - 0.3 * (t - 20) + 0.05 * h + 0.4 * d
    
    # 代入数值并计算
    milk_yield = milk_yield_expr.subs([(f, feed_quality), (t, temperature), (h, health_score), (d, drinking_time)])
    
    return float(milk_yield.evalf())

@mcp.tool
def calculate_crop_yield_by_fertilizer_symbolic(fertilizer_amount: float) -> float:
    """
    根据施肥量计算作物产量（使用sympy符号计算）
    
    公式: 产量 = 500 + 40 × x - 0.5 × x²
    其中 x 是施肥量（kg/亩）
    """
    # 定义符号变量
    x = sp.symbols('x')
    
    # 定义公式
    yield_expr = 500 + 40 * x - 0.5 * x**2
    
    # 代入数值并计算
    yield_amount = yield_expr.subs(x, fertilizer_amount)
    
    return float(yield_amount.evalf())

@mcp.tool
def calculate_ev_battery_level_symbolic(prev_battery_level: float, external_input: float) -> float:
    """
    计算电动汽车当前电池电量（使用sympy符号计算）
    
    公式: B_t = 0.95 × B_{t-1} - 1.2 + u_t
    其中 B_t 是当前电量，B_{t-1} 是前一时刻电量，u_t 是外部输入
    """
    # 定义符号变量
    B_prev, u = sp.symbols('B_prev u')
    
    # 定义公式
    battery_expr = 0.95 * B_prev - 1.2 + u
    
    # 代入数值并计算
    current_battery = battery_expr.subs([(B_prev, prev_battery_level), (u, external_input)])
    
    return float(current_battery.evalf())

@mcp.tool
def calculate_crop_growth_rate_symbolic(
    temperature: float,
    irrigation_amount: float,
    coefficient_a: float = 0.5,
    coefficient_b: float = 0.3,
    coefficient_c: float = 0.02
) -> float:
    """
    计算作物生长变化量（使用sympy符号计算）
    
    公式: ΔG_t = a×T_t + b×W_t - c×(T_t×W_t)
    其中 T_t 是平均温度，W_t 是灌溉水量
    """
    # 定义符号变量
    T, W, a, b, c = sp.symbols('T W a b c')
    
    # 定义公式
    growth_expr = a * T + b * W - c * (T * W)
    
    # 代入数值并计算
    growth_change = growth_expr.subs([
        (T, temperature), (W, irrigation_amount),
        (a, coefficient_a), (b, coefficient_b), (c, coefficient_c)
    ])
    
    return float(growth_change.evalf())

@mcp.tool
def calculate_concrete_deformation_symbolic(
    stress: float,
    elastic_modulus: float,
    time_constant: float,
    time_days: float = 365
) -> float:
    """
    计算混凝土柱长期变形（轴向应变）（使用sympy符号计算）
    
    公式: ε(t) = (σ/E) × (t - τ×ln(1 + t/τ))
    其中 σ 是轴向应力，E 是弹性模量，τ 是时间常数，t 是时间
    """
    # 定义符号变量
    sigma, E, tau, t = sp.symbols('sigma E tau t')
    
    # 定义公式
    strain_expr = (sigma / E) * (t - tau * sp.ln(1 + t / tau))
    
    # 代入数值并计算
    strain = strain_expr.subs([
        (sigma, stress), (E, elastic_modulus),
        (tau, time_constant), (t, time_days)
    ])
    
    return float(strain.evalf())

@mcp.tool
def calculate_vehicle_acceleration_symbolic(
    max_force: float,
    throttle: float,
    resistance: float,
    mass: float
) -> float:
    """
    计算车辆加速度（使用sympy符号计算）
    
    公式: a = (F - R) / m
    其中 F = F_max × throttle 是驱动力，R 是阻力，m 是质量
    """
    # 定义符号变量
    F_max, thr, R, m = sp.symbols('F_max thr R m')
    
    # 定义公式
    force = F_max * thr
    acceleration_expr = (force - R) / m
    
    # 代入数值并计算
    acceleration = acceleration_expr.subs([
        (F_max, max_force), (thr, throttle),
        (R, resistance), (m, mass)
    ])
    
    return float(acceleration.evalf())

@mcp.tool
def calculate_traffic_flow_symbolic(
    density: float,
    speed: float,
    time: float
) -> float:
    """
    计算交通流量（使用sympy符号计算）
    
    公式: Traffic Flow = density × speed × time
    """
    # 定义符号变量
    d, s, t = sp.symbols('d s t')
    
    # 定义公式
    traffic_expr = d * s * t
    
    # 代入数值并计算
    traffic_flow = traffic_expr.subs([(d, density), (s, speed), (t, time)])
    
    return float(traffic_flow.evalf())

@mcp.tool
def calculate_nonlinear_function_symbolic(x_val: float, y_val: float) -> float:
    """
    计算非线性多变量函数值（使用sympy符号计算）
    
    公式: f(x,y) = x² + 2×sin(y) + e^(0.5×x×y)
    """
    # 定义符号变量
    x, y = sp.symbols('x y')
    
    # 定义公式
    func_expr = x**2 + 2 * sp.sin(y) + sp.exp(0.5 * x * y)
    
    # 代入数值并计算
    result = func_expr.subs([(x, x_val), (y, y_val)])
    
    return float(result.evalf())

@mcp.tool
def calculate_concrete_column_deformation_symbolic(
    stress: float,
    elastic_modulus: float,
    time_constant: float,
    time_days: float = 180
) -> float:
    """
    计算混凝土柱长期变形（另一种模型）（使用sympy符号计算）
    
    公式: ε(t) = (σ/E) × (t - τ×ln(1 + t/τ))
    """
    # 定义符号变量
    sigma, E, tau, t = sp.symbols('sigma E tau t')
    
    # 定义公式
    strain_expr = (sigma / E) * (t - tau * sp.ln(1 + t / tau))
    
    # 代入数值并计算
    strain = strain_expr.subs([
        (sigma, stress), (E, elastic_modulus),
        (tau, time_constant), (t, time_days)
    ])
    
    return float(strain.evalf())

@mcp.tool
def calculate_agricultural_yield_symbolic(
    base_yield: float,
    temperature: float,
    rainfall: float,
    fertilizer: float,
    temp_threshold: float = 10,
    temp_coeff: float = 0.5,
    rain_coeff: float = 0.01,
    fert_coeff: float = 0.2
) -> float:
    """
    计算农业产量预测（使用sympy符号计算）
    
    公式: Y = Y₀ + β_T × max(T-T_base, 0) + β_R × R + β_F × F
    """
    # 定义符号变量
    Y0, T, T_base, R, F, beta_T, beta_R, beta_F = sp.symbols('Y0 T T_base R F beta_T beta_R beta_F')
    
    # 定义公式
    temp_effect = beta_T * sp.Max(T - T_base, 0)
    yield_expr = Y0 + temp_effect + beta_R * R + beta_F * F
    
    # 代入数值并计算
    yield_prediction = yield_expr.subs([
        (Y0, base_yield), (T, temperature), (T_base, temp_threshold),
        (R, rainfall), (F, fertilizer), 
        (beta_T, temp_coeff), (beta_R, rain_coeff), (beta_F, fert_coeff)
    ])
    
    return float(yield_prediction.evalf())
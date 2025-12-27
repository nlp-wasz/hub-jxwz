@mcp.tool
def temporal_simulator(t: float, a: float, b: float, c: float, d: float) -> float:
    """
    计算时间 t 时的溶解氧浓度。
    
    参数说明：
    - t: 时间
    - a: 初始DO释放量
    - b: 衰减系数
    - c: 扰动振幅
    - d: 扰动频率
    """
    exponential_term = a * math.exp(-b * t)
    sinusoidal_term = c * math.sin(d * t)
    concentration = exponential_term + sinusoidal_term
    return concentration


@mcp.tool
def daily_order_predictor(ad_spend: float, discount_rate: float, prev_orders: float) -> float:
    """
    
    预测电商当日订单量。
    
    参数说明：
    - ad_spend:      广告支出
    - discount_rate: 折扣率
    - prev_orders:   订单数
    
    """
    alpha = 0.05
    beta = 100.0
    gamma = 0.7
    
    orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
    return orders

@mcp.tool
def yield_predictor(F: float, I: float, T: float, a: float, b: float, c: float) -> float:
    """
    预测作物单位面积产量。
    
    参数说明：
    - F: 土壤肥力指数
    - I: 每周灌溉量 (mm/week)
    - T: 平均气温 (°C)
    - a, b, c: 模型经验系数

    """
    Y = a * F + b * I - c * (T ** 2)
    return Y




@mcp.tool
def food_drying_evaporation_calculator(M0: float, k: float, T: float) -> float:
    """
    
    计算累计水分蒸发量: Evap(T) = M0 * (T + (exp(-k*T) - 1) / k)
    
    """
    if k <= 0 or M0 < 0 or T < 0:
        raise ValueError("参数必须满足 M0 >= 0, k > 0, T >= 0")
    
    exp_term = math.exp(-k * T)
    evaporated = M0 * (T + (exp_term - 1) / k)
    return evaporated



@mcp.tool
def student_performance_predictor(
    study_hours: float,
    attendance_rate: float,
    quiz_avg_score: float,
    participation_score: float,
    weights: list,
    alpha: float,
    beta: float
) -> float:
    """
    
    预测学生课程成绩： mapped_participation = (participation_score - 1) * 25
    
    """
    w1, w2, w3, w4 = weights
    participation_mapped = (participation_score - 1) * 25.0
    linear_comb = (
        w1 * study_hours +
        w2 * attendance_rate +
        w3 * quiz_avg_score +
        w4 * participation_mapped
    )
    exponent = -alpha * (linear_comb - beta)
    score = 100.0 / (1.0 + math.exp(exponent))
    return score



@mcp.tool
def dynamic_system_predictor(
    x1: float, x2: float, x3: float,
    y_prev1: float, y_prev2: float,
    a: float, b: float, c: float, d: float
) -> float:
    """
    
    预测动态系统的当前输出。
    
    """
    y_t = a * x1 + b * y_prev1 + c * y_prev2 + d * x2 * x3
    return y_t



@mcp.tool
def quadratic_response_model(x: float) -> float:
    """
    计算二次函数 y = 2x^2 + 3x + 1 的输出。
    
    参数:
        x (float): 输入变量
    
    """
    return 2 * x**2 + 3 * x + 1



@mcp.tool
def media_influence_evaluator(
    content_quality: float,
    channels: int,
    engagement: float,
    time: float
) -> float:
    """
    
    计算传播项目的综合影响力

    """
    if content_quality < 0 or engagement < 0 or time < 0 or channels < 1:
        raise ValueError("输入参数必须满足非负约束，且 channels >= 1")
    
    influence = content_quality * channels * engagement * time
    return influence


@mcp.tool
def cattle_population_next_year(N_current: float, r: float, K: float) -> float:
    """
    
    预测下一年牛群数量
    
    """
    if N_current < 0 or r < 0 or K <= 0:
        raise ValueError("N_current >= 0, r >= 0, K > 0 required")
    
    N_next = N_current + r * N_current * (1 - N_current / K)
    return N_next


@mcp.tool
def weight_predictor(W0: float, r: float, t: float) -> float:
    """
    
    预测牲畜在时间 t 的体重
    
    """
    if W0 <= 0 or r < 0 or t < 0:
        raise ValueError("W0 > 0, r >= 0, t >= 0 required")
    
    W_t = W0 * math.exp(r * t)
    return W_t
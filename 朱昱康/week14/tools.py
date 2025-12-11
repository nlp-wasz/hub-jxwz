import numpy as np
import math

def get_bmr_rate(weight: float, height: float):
    """
    在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理
功能所需能量的重要指标。它广泛应用于能量需求评估、体重管理、临床营养支持等多个场
景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，
构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所
需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算。
    :param weight: 个体体重，单位为千克（kg）；
    :param height: 个体身高，单位为厘米（cm）。
    :return: 基础代谢率（BMR）
    """
    return round(10 * weight + 6.25 * height - 100, 2)

def predict_daily_sales(price_per_kg: float, supply_kg: float, demand_factor: float, 
                       discount_rate: float, weather_factor: float):
    """
    在农产品的批发与零售过程中，日销售总额受到多种因素的综合影响，包括价格、供应量、
    市场需求、促销活动以及外部环境因素如天气等。为了更精准地预测销售情况，指导采购与
    库存管理，提升经营效率，构建一个基于关键变量的定量预测模型具有重要意义。该模型旨在
    在通过量化影响因素之间的关系，提供一个可操作、可调整的销售预测工具。
    
    模型公式：
    daily_sales = price_per_kg × supply_kg × demand_factor × (1 - discount_rate) × weather_factor
    
    :param price_per_kg: 每公斤农产品的销售价格（单位：元）
    :param supply_kg: 当日可供应的农产品总量（单位：公斤）
    :param demand_factor: 需求系数，反映市场对当前农产品的需求强度，取值范围为 0 到 1
    :param discount_rate: 折扣率，表示促销活动对价格的下调比例，取值范围为 0 到 1
    :param weather_factor: 天气影响因子，用于量化天气状况对销售的影响，取值范围通常在 0.8 至 1.2 之间
    :return: 预测的日销售总额（单位：元）
    """
    # 参数验证
    if not (0 <= demand_factor <= 1):
        raise ValueError("需求系数(demand_factor)必须在0到1之间")
    if not (0 <= discount_rate <= 1):
        raise ValueError("折扣率(discount_rate)必须在0到1之间")
    if not (0.8 <= weather_factor <= 1.2):
        raise ValueError("天气影响因子(weather_factor)必须在0.8到1.2之间")
    
    # 计算日销售额
    daily_sales = price_per_kg * supply_kg * demand_factor * (1 - discount_rate) * weather_factor
    
    # 返回结果，保留两位小数
    return round(daily_sales, 2)

def calculate_monthly_rent(area: float, location_score: float, age: float, bedrooms: float, distance_to_subway: float):
    """
    在房地产投资与资产管理中，准确估算房产的潜在租金收益是评估投资回报和制定租赁策略
    的关键环节。影响租金水平的因素包括房产本身的物理属性（如面积、房龄、卧室数量）、
    地理位置（如地段评分）以及周边配套设施（如交通便利性）。为了提供一种直观且具备一
    定实用性的估算方法，构建了一个基于关键变量的代数模型，用于预测房产的预期月租金收
    益。该模型结合了线性与非线性关系，旨在反映现实中租金定价的主要驱动因素。
    
    模型公式：
    Monthly Rent = 50 × Area × (Location Score / 10) × (1 - Age / 30) × (1 + 0.1 × Bedrooms) × (1000 / (1000 + Distance to Subway))
    
    :param area: 房产面积（单位：平方米）
    :param location_score: 地段评分（1-10分）
    :param age: 房龄（单位：年）
    :param bedrooms: 卧室数量
    :param distance_to_subway: 距离地铁的距离（单位：米）
    :return: 预测的月租金（单位：元）
    """
    # 计算月租金
    monthly_rent = 50 * area * (location_score / 10) * (1 - age / 30) * (1 + 0.1 * bedrooms) * (1000 / (1000 + distance_to_subway))
    
    # 返回结果，保留两位小数
    return round(monthly_rent, 2)

def calculate_complex_system(x: float, y: float):
    """
    在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。该模型旨在
    模拟两个输入变量 x和 y对某一目标输出的综合影响，其中包含了周期性
    变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合
    作用下的信号响应机制等场景。尽管模型本身为确定性函数，但其结构设计使得输出呈现出
    类随机波动的特性，从而更好地模拟真实世界中的复杂行为。
    
    模型公式：
    fun(x, y) = 2.5 · sin(x) + 1.8 · cos(y) + 0.3 · x · y
    
    :param x: 第一个输入变量
    :param y: 第二个输入变量
    :return: 系统输出值
    """
    # 计算系统输出
    result = 2.5 * math.sin(x) + 1.8 * math.cos(y) + 0.3 * x * y
    
    # 返回结果，保留两位小数
    return round(result, 2)

def calculate_do_concentration_change(C: float, D: float, k: float, C_sat: float, bio_load: float, temp: float, x: float, t: float):
    """
    在水产养殖系统中，溶解氧（DO）是影响水生生物健康和生长的关键水质参数之一。为了
    实现对水体中溶解氧浓度的动态预测和调控，有必要建立能够反映其随时间和空间变化的数
    学模型。本模型基于扩散-反应机制，结合环境因素和生物活动的影响，构建了一个简化的
    偏微分方程框架，用于模拟溶解氧在养殖水体中的分布特征。该模型综合考虑了物理扩散、
    气体交换、温度对溶解氧饱和度的影响，以及生物代谢引起的氧消耗过程，适用于对水产养
    殖系统中的溶解氧变化趋势进行预测与分析。
    
    模型公式：
    ∂C/∂t = D·∂²C/∂x² - k(C - C_sat) + R(bio_load, temp)
    
    :param C: 当前溶解氧浓度
    :param D: 扩散系数
    :param k: 气体交换速率系数
    :param C_sat: 温度相关的溶解氧饱和浓度
    :param bio_load: 生物负载
    :param temp: 水温
    :param x: 空间位置
    :param t: 时间
    :return: 溶解氧浓度变化率
    """
    # 简化的生物耗氧速率函数
    R = 0.01 * bio_load * (1 + 0.05 * (temp - 20))
    
    # 计算溶解氧浓度变化率
    dC_dt = D * 0 - k * (C - C_sat) + R  # 简化空间二阶导数为0
    
    # 返回结果，保留四位小数
    return round(dC_dt, 4)

def calculate_milk_production(feed_quality: float, health_status: float, avg_temp: float, milk_freq: float, lactation_week: float):
    """
    在现代畜牧业中，奶牛的日均产奶量是衡量养殖场生产效率和动物健康管理质量的重要指标
    。为了实现科学饲养与精细化管理，建立一个能够反映关键影响因素的产奶量预测模型具有
    重要意义。该模型综合考虑了饲料质量、健康状况、泌乳周期、环境温度以及挤奶频率等五
    个关键变量，旨在模拟奶牛在不同饲养条件下的产奶潜力。
    
    模型设计基于确定性建模思路，即在已知输入变量的情况下，通过设定的数学关系明确计算
    出日均产奶量。该方法具有可解释性强、计算效率高、便于集成至生产管理系统等优点。通
    过此模型，养殖管理者可以评估当前饲养条件对产奶性能的影响，并据此优化饲养策略，提
    高生产效益。
    
    模型公式：
    产奶量 = 25 × (feed_quality/100) × (health_status/100) × (1 - 0.05 × |avg_temp - 20|) × (milk_freq/2) × (1 - e^(-0.1 × lactation_week))
    
    :param feed_quality: 饲料质量（0-100分）
    :param health_status: 健康状态（0-100分）
    :param avg_temp: 平均环境温度（℃）
    :param milk_freq: 每日挤奶频率
    :param lactation_week: 泌乳周数
    :return: 预测的日均产奶量（单位：kg）
    """
    # 计算产奶量
    milk_production = 25 * (feed_quality/100) * (health_status/100) * (1 - 0.05 * abs(avg_temp - 20)) * (milk_freq/2) * (1 - math.exp(-0.1 * lactation_week))
    
    # 返回结果，保留两位小数
    return round(milk_production, 2)

def calculate_complex_integral_model(x1: float, x2: float, x3: float, x4: float, x5: float, a: float, b: float, c: float, d: float, e: float):
    """
    本模型设计用于描述一个具有多变量输入的复杂系统，其输出由多种非线性机制共同决定。
    系统中包含随时间或空间累积的动态效应，通过积分形式进行建模，能够有效刻画具有衰减
    特性的累积过程。此外，模型还综合考虑了输入变量的二次效应、周期性响应、对数变换和
    幂律关系，从而增强对实际物理、工程或金融现象的拟合能力和解释性。
    
    模型公式：
    y = ∫(0 to x1) [a·e^(-t/b)/(c + t)] dt + d·x2² + e·sin(x3) + log(x4 + 1) + √x5
    
    :param x1: 积分上限变量
    :param x2: 二次项变量
    :param x3: 周期性变量
    :param x4: 对数项变量
    :param x5: 幂律项变量
    :param a: 积分项系数
    :param b: 衰减常数
    :param c: 积分分母调整参数
    :param d: 二次项系数
    :param e: 周期性项系数
    :return: 系统输出值
    """
    # 计算积分项（使用数值积分近似）
    def integrand(t):
        return a * math.exp(-t/b) / (c + t)
    
    # 简单的数值积分近似（梯形法）
    n = 100  # 分割数
    h = x1 / n
    integral = 0.5 * integrand(0) + 0.5 * integrand(x1)
    for i in range(1, n):
        integral += integrand(i * h)
    integral *= h
    
    # 计算其他项
    quadratic_term = d * x2 * x2
    periodic_term = e * math.sin(x3)
    log_term = math.log(x4 + 1)
    power_term = math.sqrt(x5) if x5 >= 0 else 0  # 确保非负
    
    # 计算总输出
    y = integral + quadratic_term + periodic_term + log_term + power_term
    
    # 返回结果，保留两位小数
    return round(y, 2)

def calculate_deterministic_model(x1: float, x2: float, x3: float, x4: float):
    """
    在实际系统中，许多问题可以通过确定性建模方法进行分析与预测。此类模型基于已知的数
    学关系，将多个输入变量通过预定义的公式转换为输出结果，广泛应用于工程、经济、物理
    等领域。为了更准确地反映变量之间的复杂关系，模型中不仅包含线性项，还引入了非线性
    成分，以增强对实际问题的拟合能力。
    
    模型公式：
    y = 2x1 + 3x2 - 1.5x3 + 0.8x4 + e^(0.1x2)
    
    :param x1: 第一个输入变量
    :param x2: 第二个输入变量
    :param x3: 第三个输入变量
    :param x4: 第四个输入变量
    :return: 系统输出值
    """
    # 计算系统输出
    y = 2 * x1 + 3 * x2 - 1.5 * x3 + 0.8 * x4 + math.exp(0.1 * x2)
    
    # 返回结果，保留两位小数
    return round(y, 2)

def calculate_do_dynamics(t: float, a: float, b: float, c: float, d: float):
    """
    在水产养殖系统中，溶解氧（Dissolved Oxygen, DO）是影响水生生物健康和生长的关键环境因子之一。
    其浓度受多种因素影响，包括水体自净能力、微生物活动、水生生物呼吸作用以及外界环境（如温度、
    光照、风力等）的周期性变化。为了更好地理解和预测DO的动态变化趋势，建立一个能够反映其非线性行为的
    数学模型具有重要意义。该模型可用于模拟封闭或半封闭养殖系统中DO浓度随时间演变的过程，为水质调控和管
    理提供理论支持。
    
    模型公式：
    DO(t) = a · e^(-b·t) + c · sin(d · t)
    
    :param t: 时间
    :param a: 初始溶解氧释放量，反映系统初始状态下的氧含量
    :param b: 溶解氧的衰减系数，刻画其随时间自然下降的速率
    :param c: 环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度
    :param d: 环境扰动的频率，反映扰动周期的快慢
    :return: 溶解氧浓度
    """
    # 计算溶解氧浓度
    do_concentration = a * math.exp(-b * t) + c * math.sin(d * t)
    
    # 返回结果，保留两位小数
    return round(do_concentration, 2)

def predict_ecommerce_orders(ad_spend: float, discount_rate: float, prev_orders: float, alpha: float = 0.05, beta: float = 100, gamma: float = 0.7):
    """
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有
    重要意义。为了构建一个简洁且具有解释性的预测模型，我们考虑从三个关键业务驱动因素
    出发：广告支出（ad_spend）、当日折扣力度（discount_rate）和前一日的订单量（prev_orders）。
    这些变量分别代表了市场推广投入、价格刺激效应以及订单增长的惯性趋势。
    
    模型公式：
    orders_t = α · ad_spend_t + β · discount_rate_t + γ · prev_orders_t
    
    :param ad_spend: 当日广告支出（单位：元）
    :param discount_rate: 当日折扣力度（0-1之间）
    :param prev_orders: 前一日订单数量
    :param alpha: 广告支出对订单量的敏感系数（默认0.05）
    :param beta: 折扣率对订单量的放大系数（默认100）
    :param gamma: 前一日订单数量对当前日订单趋势的惯性影响（默认0.7）
    :return: 预测的当日订单数量
    """
    # 计算当日订单数量
    orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
    
    # 返回结果，保留两位小数
    return round(orders, 2)

def predict_crop_yield(temp: float, rainfall: float, fertilizer: float, sunlight: float, soil_quality: float, base_yield: float = 5.0):
    """
    农业产量受到多种环境和管理因素的综合影响，为了量化这些因素对作物产量的作用，构建
    了一个简化的代数模型。该模型综合考虑了五个关键变量：平均生长温度、生长期间降水量
    、施肥量、每日平均光照时长以及土壤质量指数。通过参数组合与归一化处理，模型旨在反
    映不同条件下作物产量的变化趋势，为农业决策提供理论支持。
    
    模型公式：
    yield_prediction = base_yield × temp_factor × rainfall_factor × fertilizer_factor × sunlight_factor × soil_factor
    
    其中：
    - temp_factor = 1.0 - |temp - 25|/25，表示温度对产量的影响，以25℃为最适温度
    - rainfall_factor = 1.0 - |rainfall - 600|/600，表示降水量对产量的影响，以600mm为最优
    - fertilizer_factor = 1.0 + fertilizer/200，表示施肥量与产量的正相关关系
    - sunlight_factor = 0.8 + (sunlight/12) × 0.4，表示光照时长在8~12小时范围内对作物最为有利
    - soil_factor = 1.0 + soil_quality，表示土壤质量越高，对产量的促进作用越强
    
    :param temp: 平均生长温度（℃）
    :param rainfall: 生长期间降水量（mm）
    :param fertilizer: 施肥量（kg/公顷）
    :param sunlight: 每日平均光照时长（小时）
    :param soil_quality: 土壤质量指数（0-1之间）
    :param base_yield: 基础产量水平（默认5.0吨/公顷）
    :return: 预测的作物产量（吨/公顷）
    """
    # 计算各影响因子
    temp_factor = max(0, 1.0 - abs(temp - 25) / 25)
    rainfall_factor = max(0, 1.0 - abs(rainfall - 600) / 600)
    fertilizer_factor = 1.0 + fertilizer / 200
    sunlight_factor = 0.8 + (sunlight / 12) * 0.4
    soil_factor = 1.0 + soil_quality
    
    # 计算预测产量
    yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor
    
    # 确保产量不为负
    yield_prediction = max(0, yield_prediction)
    
    # 返回结果，保留两位小数
    return round(yield_prediction, 2)


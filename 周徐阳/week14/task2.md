## 10 个文档解析与 MCP Tool 定义

下面选取了 `documents/` 中的 10 份 PDF，提取公式并在 `tools.py` 中实现了对应的计算函数（可直接作为 MCP tool 暴露）。实现时依据需求选择了 `numpy`/`sympy` 以兼顾随机性、数值计算与积分求解。

### 1. 00470988-2938-40a2-b352-5b740b73874.pdf —— 零售销售额
- 公式：`Revenue = Input_kg * (Base_price + Fluctuation)`，波动项为区间均匀分布。
- MCP tool：`simulate_retail_revenue(input_kg, base_price, fluctuation_min, fluctuation_max, seed=None)`。
- 说明：`seed` 控制随机性复现，返回模拟的销售额。

### 2. 006ee1cd-b92f-44c1-a843-b415a035439.pdf —— 房产月租金估算
- 公式：`Monthly Rent = 50 * Area * (LocationScore/10) * (1 - Age/30) * (1 + 0.1*Bedrooms) * (1000/(1000 + Distance))`。
- MCP tool：`estimate_monthly_rent(area, location_score, age, bedrooms, distance_to_subway)`。
- 说明：对房龄、区位、卧室数、地铁距离进行线性/非线性调节。

### 3. 00ac792a-04dd-4639-abbd-d7f78cbb7ea.pdf —— 非线性交互
- 公式：`fun(x, y) = 2.5*sin(x) + 1.8*cos(y) + 0.3*x*y`。
- MCP tool：`nonlinear_interaction(x, y)`。
- 说明：结合周期项与乘积交互，输出浮点值。

### 4. 00c186c3-2266-4a12-a37b-9b740fb6a97.pdf —— 溶解氧扩散-反应
- 公式：`dC/dt = D*d2C/dx2 - k*(C - C_sat) + R(bio_load, temp)`，R 简化为 `-bio_load * (1 + temp_sensitivity*(temp-20))`。
- MCP tool：`dissolved_oxygen_rate(concentration, second_derivative, diffusion_coeff, exchange_rate, saturation_conc, bio_load, temp_c, temp_sensitivity=0.02)`。
- 说明：传入二阶导近似值，返回溶解氧变化率。

### 5. 01867e47-0307-4492-bb14-b02831ac877.pdf —— 库存衰减与补货
- 公式：`dS/dt = -k*S + u(t)`，显式欧拉：`S_next = S + (-k*S + u)*dt`。
- MCP tool：`inventory_step(current_stock, decay_rate, replenishment, dt=1.0)`。
- 说明：模拟单步库存变化，结果下限为 0。

### 6. 02c197ad-d1b7-4453-abda-96c335a7fe0.pdf —— 奶牛日均产奶量
- 公式：`25*(fq/100)*(hs/100)*(1 - 0.05*|temp-20|)*(milk_freq/2)*(1 - exp(-0.1*lactation_week))`。
- MCP tool：`predict_milk_yield(feed_quality, health_status, avg_temp, milk_freq, lactation_week)`。
- 说明：温度与泌乳周期项做下限截断，避免负产奶量。

### 7. 02e55900-9b81-44f2-b1de-5210365bc87.pdf —— 复合积分与非线性响应
- 公式：`y = ∫_0^{x1} a*exp(-t/b)/(c+t) dt + d*x2^2 + e*sin(x3) + log(x4+1) + sqrt(x5)`。
- MCP tool：`complex_system_response(x1, x2, x3, x4, x5, a, b, c, d, e)`。
- 说明：积分由 `sympy` 计算，对对数/平方根项做非负保护。

### 8. 03076ad4-a83c-4672-89a8-8245b7a2443.pdf —— 体重差分模型
- 公式：`W_{t+1} = W_t + (C_t - E_t)/k`。
- MCP tool：`next_body_weight(current_weight, calorie_intake, calorie_expenditure, k)`。
- 说明：`k` 为热量-体重换算系数，需非零。

### 9. 04744ec2-0239-42f1-bdd4-52a626bd744.pdf —— 简化衍生品定价
- 公式：`price = exp(-r*T) * max(S - K, 0)`，等价于 `max(S*e^{-rT} - K*e^{-rT}, 0)`。
- MCP tool：`discounted_derivative_price(s, k, r, t)`。
- 说明：体现贴现与非负性，可用于简化的 PDE 定价演示。

### 10. 05249a97-560d-49c6-b646-2454129f157.pdf —— 日增重（ADG）
- 公式：`ADG = (feed_intake * protein_content) / (animal_weight * 10)`。
- MCP tool：`average_daily_gain(feed_intake, protein_content, animal_weight)`。
- 说明：体重需为正，否则抛出异常；返回单位为 kg/天。

### 使用说明
- 实现位置：`tools.py`，可直接按函数名暴露为 MCP 工具。
- 依赖：`numpy`（随机、三角函数等），`sympy`（定积分求解），标准库 `math`。
- 校验建议：为含随机性的 `simulate_retail_revenue` 设置固定 `seed` 以便复现；对价格/库存等业务量可按需增加输入约束。

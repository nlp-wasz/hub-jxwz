import numpy as np
import math
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Simplified PDE Option Pricing Model")

@mcp.tool()
def calculate_simplified_pde_price(
    S: float = 100.0,    # 标的资产价格 (Spot price)
    K: float = 100.0,    # 行权价格 (Strike price)
    T: float = 1.0,      # 到期时间（年）(Time to maturity in years)
    r: float = 0.05      # 无风险利率 (Risk-free interest rate)
) -> float:
    """
    基于简化PDE模型的金融衍生品定价计算
    
    建模背景:
    在金融工程与衍生品定价领域，偏微分方程（PDE）是构建动态定价模型的核心工具之一。
    Black-Scholes-Merton模型作为经典范例，奠定了欧式期权定价的理论基础。
    该模型通过对构造标的资产价格和时间的连续函数、结合无套利原理，推导出描述
    期权的概率化的偏微分方程。尽管实际应用中常采用解析解或数值方法求解该方程，
    但在教学和初步分析中，简化模型有助于理解PDE建模的基本逻辑。
    
    建模公式:
    \[ \text{price} = \max (Se^{-rT} - Ke^{-rT}, 0) \]
    
    该表达式表示一种简化的衍生品定价形式，其中包含了对资产价格的指数贴现机制，
    并确保输出非负，模拟了期权等金融工具的价格特性。
    
    参数说明:
    - S: 标的资产价格 (Spot price)
    - K: 行权价格 (Strike price)
    - T: 到期时间，以年为单位 (Time to maturity in years)
    - r: 无风险利率 (Risk-free interest rate)
    
    返回:
    - price: 衍生品理论价格 (Theoretical price of the derivative)
    
    使用示例:
    >>> calculate_simplified_pde_price(S=110, K=100, T=1.0, r=0.05)
    9.512294
    """
    # 计算贴现因子
    discount_factor = math.exp(-r * T)
    
    # 按照PDF中的公式计算：max(Se^{-rT} - Ke^{-rT}, 0)
    discounted_S = S * discount_factor
    discounted_K = K * discount_factor
    price = max(discounted_S - discounted_K, 0)
    
    return price

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
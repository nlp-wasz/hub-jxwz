import numpy as np
from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Livestock ADG Calculator")

@mcp.tool()
def calculate_adg(
    feed_intake: float = 5.0,        # 日采食量 (kg/天)
    protein_content: float = 16.0,   # 饲料粗蛋白含量 (%)
    animal_weight: float = 200.0     # 动物当前体重 (kg)
) -> float:
    """
    畜牧业日增重（ADG）计算模型
    
    建模背景:
    在畜牧业生产中，日增重（Average Daily Gain, ADG）是评估动物生长性能的重要指标之一。
    为了科学地预测和管理牲畜的生长速度，建立一个基于关键营养和生理参数的确定性模型具有重要意义。
    该模型有助于理解饲料摄入、饲料质量与动物体重之间的相互作用，并为饲养管理决策提供理论支持。
    
    本模型选取了三个关键变量：日采食量（feed_intake）、饲料粗蛋白含量（protein_content）
    以及动物当前体重（animal_weight）。日采食量反映了动物的摄食能力，粗蛋白含量是影响
    动物生长效率的重要营养因子，而动物体重则作为生长阶段的生理指标，对增重效率具有调节作用。
    
    建模公式:
    \[ADG = \frac{feed\_intake \times protein\_content}{animal\_weight \times 10}\]
    
    公式说明:
    该公式中，ADG表示每日平均增重（单位：kg/天），模型假设增重与日采食量和粗蛋白含量
    呈正相关，与动物当前体重呈负相关，体现了体重增长边际效率递减的趋势。该模型可用于
    不同饲养条件下牲畜生长性能的初步预测与比较分析。
    
    参数说明:
    - feed_intake: 日采食量，单位：kg/天 (Daily feed intake in kg/day)
    - protein_content: 饲料粗蛋白含量，单位：% (Crude protein content in percentage)
    - animal_weight: 动物当前体重，单位：kg (Current animal weight in kg)
    
    返回:
    - ADG: 日增重，单位：kg/天 (Average Daily Gain in kg/day)
    
    使用示例:
    >>> calculate_adg(feed_intake=5.0, protein_content=16.0, animal_weight=200.0)
    0.04
    """
    # 检查输入参数的合理性
    if feed_intake <= 0:
        raise ValueError("日采食量必须为正数")
    if protein_content <= 0:
        raise ValueError("粗蛋白含量必须为正数")
    if animal_weight <= 0:
        raise ValueError("动物体重必须为正数")
    
    # 按照PDF中的公式计算：ADG = (feed_intake × protein_content) / (animal_weight × 10)
    adg = (feed_intake * protein_content) / (animal_weight * 10)
    
    return adg

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
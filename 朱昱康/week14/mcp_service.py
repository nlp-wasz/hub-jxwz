#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP服务定义模块
将每个工具函数定义为MCP可执行的计算工程
"""

import json
from typing import Dict, Any, List, Optional
import sys
import os

# 添加tools.py所在目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import (
    get_bmr_rate,
    predict_daily_sales,
    calculate_monthly_rent,
    calculate_complex_system,
    calculate_do_concentration_change,
    calculate_milk_production,
    calculate_complex_integral_model,
    calculate_deterministic_model,
    calculate_do_dynamics,
    predict_ecommerce_orders,
    predict_crop_yield
)

class MCPService:
    """MCP服务类，用于定义和执行工具函数"""
    
    def __init__(self):
        """初始化MCP服务"""
        # 工具函数映射
        self.tools = {
            "get_bmr_rate": {
                "name": "基础代谢率计算",
                "description": "计算个体在静息状态下维持基本生理功能所需能量的重要指标",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight": {
                            "type": "number",
                            "description": "个体体重，单位为千克（kg）"
                        },
                        "height": {
                            "type": "number",
                            "description": "个体身高，单位为厘米（cm）"
                        }
                    },
                    "required": ["weight", "height"]
                },
                "function": get_bmr_rate
            },
            "predict_daily_sales": {
                "name": "农产品日销售额预测",
                "description": "预测农产品的日销售总额，考虑价格、供应量、市场需求、促销活动以及天气等因素的影响",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price_per_kg": {
                            "type": "number",
                            "description": "每公斤农产品的销售价格（单位：元）"
                        },
                        "supply_kg": {
                            "type": "number",
                            "description": "当日可供应的农产品总量（单位：公斤）"
                        },
                        "demand_factor": {
                            "type": "number",
                            "description": "需求系数，反映市场对当前农产品的需求强度，取值范围为 0 到 1"
                        },
                        "discount_rate": {
                            "type": "number",
                            "description": "折扣率，表示促销活动对价格的下调比例，取值范围为 0 到 1"
                        },
                        "weather_factor": {
                            "type": "number",
                            "description": "天气影响因子，用于量化天气状况对销售的影响，取值范围通常在 0.8 至 1.2 之间"
                        }
                    },
                    "required": ["price_per_kg", "supply_kg", "demand_factor", "discount_rate", "weather_factor"]
                },
                "function": predict_daily_sales
            },
            "calculate_monthly_rent": {
                "name": "房产月租金预测",
                "description": "预测房产的潜在月租金收益，考虑面积、地段评分、房龄、卧室数量和交通便利性等因素",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {
                            "type": "number",
                            "description": "房产面积（单位：平方米）"
                        },
                        "location_score": {
                            "type": "number",
                            "description": "地段评分（1-10分）"
                        },
                        "age": {
                            "type": "number",
                            "description": "房龄（单位：年）"
                        },
                        "bedrooms": {
                            "type": "number",
                            "description": "卧室数量"
                        },
                        "distance_to_subway": {
                            "type": "number",
                            "description": "距离地铁的距离（单位：米）"
                        }
                    },
                    "required": ["area", "location_score", "age", "bedrooms", "distance_to_subway"]
                },
                "function": calculate_monthly_rent
            },
            "calculate_complex_system": {
                "name": "复杂系统分析",
                "description": "模拟两个输入变量对某一目标输出的综合影响，包含周期性变化与线性交互的成分",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "第一个输入变量"
                        },
                        "y": {
                            "type": "number",
                            "description": "第二个输入变量"
                        }
                    },
                    "required": ["x", "y"]
                },
                "function": calculate_complex_system
            },
            "calculate_do_concentration_change": {
                "name": "溶解氧浓度变化计算",
                "description": "基于扩散-反应机制，预测水产养殖系统中溶解氧浓度的动态变化",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "C": {
                            "type": "number",
                            "description": "当前溶解氧浓度"
                        },
                        "D": {
                            "type": "number",
                            "description": "扩散系数"
                        },
                        "k": {
                            "type": "number",
                            "description": "气体交换速率系数"
                        },
                        "C_sat": {
                            "type": "number",
                            "description": "温度相关的溶解氧饱和浓度"
                        },
                        "bio_load": {
                            "type": "number",
                            "description": "生物负载"
                        },
                        "temp": {
                            "type": "number",
                            "description": "水温"
                        },
                        "x": {
                            "type": "number",
                            "description": "空间位置"
                        },
                        "t": {
                            "type": "number",
                            "description": "时间"
                        }
                    },
                    "required": ["C", "D", "k", "C_sat", "bio_load", "temp", "x", "t"]
                },
                "function": calculate_do_concentration_change
            },
            "calculate_milk_production": {
                "name": "奶牛产奶量预测",
                "description": "预测奶牛的日均产奶量，考虑饲料质量、健康状况、泌乳周期、环境温度和挤奶频率等因素",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feed_quality": {
                            "type": "number",
                            "description": "饲料质量（0-100分）"
                        },
                        "health_status": {
                            "type": "number",
                            "description": "健康状态（0-100分）"
                        },
                        "avg_temp": {
                            "type": "number",
                            "description": "平均环境温度（℃）"
                        },
                        "milk_freq": {
                            "type": "number",
                            "description": "每日挤奶频率"
                        },
                        "lactation_week": {
                            "type": "number",
                            "description": "泌乳周数"
                        }
                    },
                    "required": ["feed_quality", "health_status", "avg_temp", "milk_freq", "lactation_week"]
                },
                "function": calculate_milk_production
            },
            "calculate_complex_integral_model": {
                "name": "复杂积分模型计算",
                "description": "描述多变量输入的复杂系统，包含随时间或空间累积的动态效应",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                            "type": "number",
                            "description": "积分上限变量"
                        },
                        "x2": {
                            "type": "number",
                            "description": "二次项变量"
                        },
                        "x3": {
                            "type": "number",
                            "description": "周期性变量"
                        },
                        "x4": {
                            "type": "number",
                            "description": "对数项变量"
                        },
                        "x5": {
                            "type": "number",
                            "description": "幂律项变量"
                        },
                        "a": {
                            "type": "number",
                            "description": "积分项系数"
                        },
                        "b": {
                            "type": "number",
                            "description": "衰减常数"
                        },
                        "c": {
                            "type": "number",
                            "description": "积分分母调整参数"
                        },
                        "d": {
                            "type": "number",
                            "description": "二次项系数"
                        },
                        "e": {
                            "type": "number",
                            "description": "周期性项系数"
                        }
                    },
                    "required": ["x1", "x2", "x3", "x4", "x5", "a", "b", "c", "d", "e"]
                },
                "function": calculate_complex_integral_model
            },
            "calculate_deterministic_model": {
                "name": "确定性模型计算",
                "description": "基于已知数学关系，将多个输入变量通过预定义公式转换为输出结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                            "type": "number",
                            "description": "第一个输入变量"
                        },
                        "x2": {
                            "type": "number",
                            "description": "第二个输入变量"
                        },
                        "x3": {
                            "type": "number",
                            "description": "第三个输入变量"
                        },
                        "x4": {
                            "type": "number",
                            "description": "第四个输入变量"
                        }
                    },
                    "required": ["x1", "x2", "x3", "x4"]
                },
                "function": calculate_deterministic_model
            },
            "calculate_do_dynamics": {
                "name": "溶解氧动态变化",
                "description": "模拟水产养殖系统中溶解氧浓度随时间演变的过程",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "t": {
                            "type": "number",
                            "description": "时间"
                        },
                        "a": {
                            "type": "number",
                            "description": "初始溶解氧释放量，反映系统初始状态下的氧含量"
                        },
                        "b": {
                            "type": "number",
                            "description": "溶解氧的衰减系数，刻画其随时间自然下降的速率"
                        },
                        "c": {
                            "type": "number",
                            "description": "环境扰动的振幅，体现外部周期性因素对DO浓度的影响强度"
                        },
                        "d": {
                            "type": "number",
                            "description": "环境扰动的频率，反映扰动周期的快慢"
                        }
                    },
                    "required": ["t", "a", "b", "c", "d"]
                },
                "function": calculate_do_dynamics
            },
            "predict_ecommerce_orders": {
                "name": "电商订单预测",
                "description": "预测每日订单增长量，考虑广告支出、折扣力度和前一日的订单量等因素",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ad_spend": {
                            "type": "number",
                            "description": "当日广告支出（单位：元）"
                        },
                        "discount_rate": {
                            "type": "number",
                            "description": "当日折扣力度（0-1之间）"
                        },
                        "prev_orders": {
                            "type": "number",
                            "description": "前一日的订单数量"
                        },
                        "alpha": {
                            "type": "number",
                            "description": "广告支出对订单量的敏感系数（默认0.05）"
                        },
                        "beta": {
                            "type": "number",
                            "description": "折扣率对订单量的放大系数（默认100）"
                        },
                        "gamma": {
                            "type": "number",
                            "description": "前一日的订单数量对当前日订单趋势的惯性影响（默认0.7）"
                        }
                    },
                    "required": ["ad_spend", "discount_rate", "prev_orders"]
                },
                "function": predict_ecommerce_orders
            },
            "predict_crop_yield": {
                "name": "农作物产量预测",
                "description": "预测农作物产量，考虑温度、降水量、施肥量、光照时长和土壤质量等因素",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "temp": {
                            "type": "number",
                            "description": "平均生长温度（℃）"
                        },
                        "rainfall": {
                            "type": "number",
                            "description": "生长期间降水量（mm）"
                        },
                        "fertilizer": {
                            "type": "number",
                            "description": "施肥量（kg/公顷）"
                        },
                        "sunlight": {
                            "type": "number",
                            "description": "每日平均光照时长（小时）"
                        },
                        "soil_quality": {
                            "type": "number",
                            "description": "土壤质量指数（0-1之间）"
                        },
                        "base_yield": {
                            "type": "number",
                            "description": "基础产量水平（默认5.0吨/公顷）"
                        }
                    },
                    "required": ["temp", "rainfall", "fertilizer", "sunlight", "soil_quality"]
                },
                "function": predict_crop_yield
            }
        }
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """
        获取所有可用工具的列表
        
        Returns:
            List[Dict[str, Any]]: 工具列表
        """
        tool_list = []
        for tool_id, tool_info in self.tools.items():
            tool_list.append({
                "id": tool_id,
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            })
        return tool_list
    
    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定的工具函数
        
        Args:
            tool_id (str): 工具ID
            parameters (Dict[str, Any]): 参数字典
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if tool_id not in self.tools:
            return {
                "success": False,
                "error": f"工具 '{tool_id}' 不存在"
            }
        
        try:
            tool_info = self.tools[tool_id]
            function = tool_info["function"]
            
            # 执行函数
            result = function(**parameters)
            
            return {
                "success": True,
                "result": result,
                "tool_id": tool_id,
                "tool_name": tool_info["name"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id,
                "tool_name": tool_info["name"] if tool_id in self.tools else "Unknown"
            }
    
    def get_tool_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定工具的参数模式
        
        Args:
            tool_id (str): 工具ID
            
        Returns:
            Optional[Dict[str, Any]]: 工具参数模式
        """
        if tool_id not in self.tools:
            return None
        
        return self.tools[tool_id]["parameters"]
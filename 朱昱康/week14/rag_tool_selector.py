#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG工具筛选模块
用于计算用户问题与工具函数的相似度，选择最相关的工具函数
"""

import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ToolSelector:
    """工具选择器，基于RAG方法选择最相关的工具函数"""
    
    def __init__(self):
        """初始化工具选择器"""
        # 工具函数描述字典
        self.tools = {
            "get_bmr_rate": {
                "name": "基础代谢率计算",
                "description": "计算个体在静息状态下维持基本生理功能所需能量的重要指标，基于体重和身高数据快速估算其每日基础代谢所需热量",
                "keywords": ["基础代谢率", "BMR", "代谢", "能量需求", "体重", "身高", "健康", "营养评估"],
                "params": ["weight", "height"]
            },
            "predict_daily_sales": {
                "name": "农产品日销售额预测",
                "description": "预测农产品的日销售总额，考虑价格、供应量、市场需求、促销活动以及天气等因素的影响",
                "keywords": ["农产品", "销售", "预测", "价格", "供应量", "需求", "促销", "天气", "批发", "零售"],
                "params": ["price_per_kg", "supply_kg", "demand_factor", "discount_rate", "weather_factor"]
            },
            "calculate_monthly_rent": {
                "name": "房产月租金预测",
                "description": "预测房产的潜在月租金收益，考虑面积、地段评分、房龄、卧室数量和交通便利性等因素",
                "keywords": ["房产", "租金", "预测", "面积", "地段", "房龄", "卧室", "地铁", "交通便利"],
                "params": ["area", "location_score", "age", "bedrooms", "distance_to_subway"]
            },
            "calculate_complex_system": {
                "name": "复杂系统分析",
                "description": "模拟两个输入变量对某一目标输出的综合影响，包含周期性变化与线性交互的成分",
                "keywords": ["复杂系统", "非线性", "交互", "周期性", "变量", "系统响应"],
                "params": ["x", "y"]
            },
            "calculate_do_concentration_change": {
                "name": "溶解氧浓度变化计算",
                "description": "基于扩散-反应机制，预测水产养殖系统中溶解氧浓度的动态变化",
                "keywords": ["溶解氧", "DO", "水产养殖", "水质", "扩散", "反应", "浓度变化"],
                "params": ["C", "D", "k", "C_sat", "bio_load", "temp", "x", "t"]
            },
            "calculate_milk_production": {
                "name": "奶牛产奶量预测",
                "description": "预测奶牛的日均产奶量，考虑饲料质量、健康状况、泌乳周期、环境温度和挤奶频率等因素",
                "keywords": ["奶牛", "产奶量", "畜牧业", "饲料", "健康状况", "泌乳周期", "温度", "挤奶频率"],
                "params": ["feed_quality", "health_status", "avg_temp", "milk_freq", "lactation_week"]
            },
            "calculate_complex_integral_model": {
                "name": "复杂积分模型计算",
                "description": "描述多变量输入的复杂系统，包含随时间或空间累积的动态效应",
                "keywords": ["积分", "复杂系统", "多变量", "非线性", "累积", "动态效应"],
                "params": ["x1", "x2", "x3", "x4", "x5", "a", "b", "c", "d", "e"]
            },
            "calculate_deterministic_model": {
                "name": "确定性模型计算",
                "description": "基于已知数学关系，将多个输入变量通过预定义公式转换为输出结果",
                "keywords": ["确定性", "线性", "非线性", "模型", "变量", "输出"],
                "params": ["x1", "x2", "x3", "x4"]
            },
            "calculate_do_dynamics": {
                "name": "溶解氧动态变化",
                "description": "模拟水产养殖系统中溶解氧浓度随时间演变的过程",
                "keywords": ["溶解氧", "DO", "水产养殖", "动态变化", "时间演变", "水质调控"],
                "params": ["t", "a", "b", "c", "d"]
            },
            "predict_ecommerce_orders": {
                "name": "电商订单预测",
                "description": "预测每日订单增长量，考虑广告支出、折扣力度和前一日的订单量等因素",
                "keywords": ["电商", "订单", "预测", "广告支出", "折扣", "订单增长"],
                "params": ["ad_spend", "discount_rate", "prev_orders", "alpha", "beta", "gamma"]
            },
            "predict_crop_yield": {
                "name": "农作物产量预测",
                "description": "预测农作物产量，考虑温度、降水量、施肥量、光照时长和土壤质量等因素",
                "keywords": ["农业", "作物", "产量", "温度", "降水", "施肥", "光照", "土壤质量"],
                "params": ["temp", "rainfall", "fertilizer", "sunlight", "soil_quality", "base_yield"]
            }
        }
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)),
            min_df=1,
            max_df=0.8,
            max_features=1000
        )
        
        # 构建工具描述的语料库
        corpus = []
        for tool_id, tool_info in self.tools.items():
            # 组合名称、描述和关键词作为工具的文本表示
            text = f"{tool_info['name']} {tool_info['description']} {' '.join(tool_info['keywords'])}"
            corpus.append(text)
        
        # 训练TF-IDF模型
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def select_tool(self, query: str, top_k: int = 3):
        """
        根据用户查询选择最相关的工具函数
        
        Args:
            query (str): 用户查询
            top_k (int): 返回的最相关工具数量
            
        Returns:
            list: 包含工具ID和相似度分数的列表
        """
        # 对查询进行分词和向量化
        query_vec = self.vectorizer.transform([query])
        
        # 计算查询与所有工具的余弦相似度
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 获取相似度最高的top_k个工具
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            tool_id = list(self.tools.keys())[idx]
            tool_info = self.tools[tool_id]
            results.append({
                "tool_id": tool_id,
                "name": tool_info["name"],
                "description": tool_info["description"],
                "similarity": float(similarities[idx]),
                "params": tool_info["params"]
            })
        
        return results
    
    def extract_parameters(self, query: str, tool_id: str):
        """
        从用户查询中提取工具函数所需的参数
        
        Args:
            query (str): 用户查询
            tool_id (str): 工具ID
            
        Returns:
            dict: 提取的参数字典
        """
        if tool_id not in self.tools:
            return {}
        
        tool_info = self.tools[tool_id]
        params = tool_info["params"]
        result = {}
        
        # 尝试从查询中提取数值参数
        for param in params:
            # 使用正则表达式尝试提取参数值
            patterns = [
                rf"{param}[：:=]\s*([0-9]+\.?[0-9]*)",  # param: 123 或 param=123
                rf"{param}\s*[是为是]\s*([0-9]+\.?[0-9]*)",  # param是123
                rf"([0-9]+\.?[0-9]*)\s*[是为是]\s*{param}",  # 123是param
                rf"([0-9]+\.?[0-9]*)\s*的{param}",  # 123的param
                rf"{param}\s*([0-9]+\.?[0-9]*)",  # param 123
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    try:
                        value = float(match.group(1))
                        result[param] = value
                        break
                    except ValueError:
                        continue
        
        return result
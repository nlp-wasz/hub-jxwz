from sentence_transformers import SentenceTransformer, util
import torch

model_name = '../Qwen3-Embedding' # qwen3 embedding， 和bge 相比更加强大，但速度更慢。

class Tool_Selector:
    """工具选择器"""
    def __init__(self):
        self.tools={
            "get_bmr_rate":"""在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理功能所需能量的重要指标。
            它广泛应用于能量需求评估、体重管理、临床营养支持等多个场景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，
                            构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所
                            需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算.""",
            "calculate_complex_system":""" 在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。该模型旨在
                            模拟两个输入变量 x和 y对某一目标输出的综合影响，其中包含了周期性
                            变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合
                            作用下的信号响应机制等场景。尽管模型本身为确定性函数，但其结构设计使得输出呈现出
                            类随机波动的特性，从而更好地模拟真实世界中的复杂行为。""",
            "calculate_deterministic_model":"""在实际系统中，许多问题可以通过确定性建模方法进行分析与预测。此类模型基于已知的数
                            学关系，将多个输入变量通过预定义的公式转换为输出结果，广泛应用于工程、经济、物理
                            等领域。为了更准确地反映变量之间的复杂关系，模型中不仅包含线性项，还引入了非线性
                            成分，以增强对实际问题的拟合能力。""",
            "calculate_monthly_rent":"""在房地产投资与资产管理中，准确估算房产的潜在租金收益是评估投资回报和制定租赁策略
                            的关键环节。影响租金水平的因素包括房产本身的物理属性（如面积、房龄、卧室数量）、
                            地理位置（如地段评分）以及周边配套设施（如交通便利性）。为了提供一种直观且具备一
                            定实用性的估算方法，构建了一个基于关键变量的代数模型，用于预测房产的预期月租金收
                            益。该模型结合了线性与非线性关系，旨在反映现实中租金定价的主要驱动因素。""",
            "calculate_do_concentration_change":""" 在水产养殖系统中，溶解氧（DO）是影响水生生物健康和生长的关键水质参数之一。为了
                            实现对水体中溶解氧浓度的动态预测和调控，有必要建立能够反映其随时间和空间变化的数
                            学模型。本模型基于扩散-反应机制，结合环境因素和生物活动的影响，构建了一个简化的
                            偏微分方程框架，用于模拟溶解氧在养殖水体中的分布特征。该模型综合考虑了物理扩散、
                            气体交换、温度对溶解氧饱和度的影响，以及生物代谢引起的氧消耗过程，适用于对水产养
                            殖系统中的溶解氧变化趋势进行预测与分析。""",
            "calculate_milk_production":"""在现代畜牧业中，奶牛的日均产奶量是衡量养殖场生产效率和动物健康管理质量的重要指标
                            。为了实现科学饲养与精细化管理，建立一个能够反映关键影响因素的产奶量预测模型具有
                            重要意义。该模型综合考虑了饲料质量、健康状况、泌乳周期、环境温度以及挤奶频率等五
                            个关键变量，旨在模拟奶牛在不同饲养条件下的产奶潜力。
                            
                            模型设计基于确定性建模思路，即在已知输入变量的情况下，通过设定的数学关系明确计算
                            出日均产奶量。该方法具有可解释性强、计算效率高、便于集成至生产管理系统等优点。通
                            过此模型，养殖管理者可以评估当前饲养条件对产奶性能的影响，并据此优化饲养策略，提
                            高生产效益。""",
            "predict_crop_yield":"""农业产量受到多种环境和管理因素的综合影响，为了量化这些因素对作物产量的作用，构建
                            了一个简化的代数模型。该模型综合考虑了五个关键变量：平均生长温度、生长期间降水量
                            、施肥量、每日平均光照时长以及土壤质量指数。通过参数组合与归一化处理，模型旨在反
                            映不同条件下作物产量的变化趋势，为农业决策提供理论支持。""",
            "predict_daily_sales":""" 在农产品的批发与零售过程中，日销售总额受到多种因素的综合影响，包括价格、供应量、
                            市场需求、促销活动以及外部环境因素如天气等。为了更精准地预测销售情况，指导采购与
                            库存管理，提升经营效率，构建一个基于关键变量的定量预测模型具有重要意义。该模型旨在
                            在通过量化影响因素之间的关系，提供一个可操作、可调整的销售预测工具。"""
        }
        self.embedding= SentenceTransformer(model_name, trust_remote_code=True)
        #构建公式的语料库
        contents = []
        for tool_name,tool_description in self.tools.items():
            contents.append(tool_description)
        self.tool_embeddings = self.embedding.encode_document(contents,convert_to_tensor=True)  # <class 'numpy.ndarray'>

    def get_similarity_tool(self,query,topk):
        query_embedding=self.embedding.encode_query(query,convert_to_tensor=True)
        similarity_score=self.embedding.similarity(query_embedding,self.tool_embeddings)[0]
        top_scores, top_indices = torch.topk(similarity_score, k=topk)
        tool = []
        for indice in top_indices:
            tool.append(list(tools.keys())[indice])

        return tool





tools = {
    "get_bmr_rate":  "在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理"
                      "功能所需能量的重要指标。它广泛应用于能量需求评估、体重管理、临床营养支持等多个场"
                      "景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，"
                       "构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所"
                        "需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算.",
    "calculate_complex_system":"在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。该模型旨在"
                            "模拟两个输入变量 x和 y对某一目标输出的综合影响，其中包含了周期性"
                            "变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合"
                            "作用下的信号响应机制等场景。尽管模型本身为确定性函数，但其结构设计使得输出呈现出"
                            "类随机波动的特性，从而更好地模拟真实世界中的复杂行为。",
    "calculate_deterministic_model": "在实际系统中，许多问题可以通过确定性建模方法进行分析与预测。此类模型基于已知的数"
                            "学关系，将多个输入变量通过预定义的公式转换为输出结果，广泛应用于工程、经济、物理"
                            "等领域。为了更准确地反映变量之间的复杂关系，模型中不仅包含线性项，还引入了非线性"
                            "成分，以增强对实际问题的拟合能力。",
    "calculate_monthly_rent": "在房地产投资与资产管理中，准确估算房产的潜在租金收益是评估投资回报和制定租赁策略"
                            "的关键环节。影响租金水平的因素包括房产本身的物理属性（如面积、房龄、卧室数量）、"
                            "地理位置（如地段评分）以及周边配套设施（如交通便利性）。为了提供一种直观且具备一"
                            "定实用性的估算方法，构建了一个基于关键变量的代数模型，用于预测房产的预期月租金收"
                           " 益。该模型结合了线性与非线性关系，旨在反映现实中租金定价的主要驱动因素。",
    "calculate_do_concentration_change": "在水产养殖系统中，溶解氧（DO）是影响水生生物健康和生长的关键水质参数之一。为了"
                            "实现对水体中溶解氧浓度的动态预测和调控，有必要建立能够反映其随时间和空间变化的数"
                            "学模型。本模型基于扩散-反应机制，结合环境因素和生物活动的影响，构建了一个简化的"
                            "偏微分方程框架，用于模拟溶解氧在养殖水体中的分布特征。该模型综合考虑了物理扩散、"
                            "气体交换、温度对溶解氧饱和度的影响，以及生物代谢引起的氧消耗过程，适用于对水产养"
                            "殖系统中的溶解氧变化趋势进行预测与分析。",
    "calculate_milk_production":" 在现代畜牧业中，奶牛的日均产奶量是衡量养殖场生产效率和动物健康管理质量的重要指标"
                           " 。为了实现科学饲养与精细化管理，建立一个能够反映关键影响因素的产奶量预测模型具有"
                            "重要意义。该模型综合考虑了饲料质量、健康状况、泌乳周期、环境温度以及挤奶频率等五"
                            "个关键变量，旨在模拟奶牛在不同饲养条件下的产奶潜力。"
                            "模型设计基于确定性建模思路，即在已知输入变量的情况下，通过设定的数学关系明确计算"
                            "出日均产奶量。该方法具有可解释性强、计算效率高、便于集成至生产管理系统等优点。通"
                            "过此模型，养殖管理者可以评估当前饲养条件对产奶性能的影响，并据此优化饲养策略，提"
                            "高生产效益。",
    "predict_crop_yield": "农业产量受到多种环境和管理因素的综合影响，为了量化这些因素对作物产量的作用，构建"
                            "了一个简化的代数模型。该模型综合考虑了五个关键变量：平均生长温度、生长期间降水量"
                            "、施肥量、每日平均光照时长以及土壤质量指数。通过参数组合与归一化处理，模型旨在反"
                           " 映不同条件下作物产量的变化趋势，为农业决策提供理论支持。",
    "predict_daily_sales": "在农产品的批发与零售过程中，日销售总额受到多种因素的综合影响，包括价格、供应量、"
                            "市场需求、促销活动以及外部环境因素如天气等。为了更精准地预测销售情况，指导采购与"
                            "库存管理，提升经营效率，构建一个基于关键变量的定量预测模型具有重要意义。该模型旨在"
                            "在通过量化影响因素之间的关系，提供一个可操作、可调整的销售预测工具。"
}
embedding = SentenceTransformer(model_name, trust_remote_code=True)
# 构建公式的语料库
contents = []
for tool_name, tool_description in tools.items():
    tool_description = tool_description.replace('\n', ' ')
    contents.append(tool_description)

query="身高163cm,体重110斤，体脂率是多少？"
tool_embeddings=embedding.encode_document(contents,convert_to_tensor=True) #转成tensor能在GPU上加速运算
query_embedding=embedding.encode_query(query,convert_to_tensor=True)
similarity_score=embedding.similarity(query_embedding,tool_embeddings)[0]
print(similarity_score)
top_scores, top_indices = torch.topk(similarity_score, k=3)
print(top_scores,top_indices)

tool=[]
for indice in top_indices:
    tool.append(list(tools.keys())[indice])
print(tool)






        


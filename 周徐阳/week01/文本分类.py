import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归
from sklearn.tree import DecisionTreeClassifier      # 导入决策树

# 1.KNN
# 读取训练数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 中文分词处理
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 文本向量化
# 词汇表: ["帮我", "导航", "到", "南京路", "音乐", "天气"]
# "帮我 导航 到 南京路" → [1, 1, 1, 1, 0, 0]  (前4个词各出现1次)
# "播放 音乐"           → [0, 0, 0, 0, 1, 0]  (只有"音乐"出现1次)
vector = CountVectorizer()                      # 创建向量化工具
vector.fit(input_sententce.values)            # 学习词汇表
input_feature = vector.transform(input_sententce.values)  # 转换为数字特征

# 训练机器学习模型
# 使用训练数据教会模型识别不同类型的文本
# input_feature：文本的数字特征
# dataset[1].values：对应的分类标签
model = KNeighborsClassifier()                  # 创建KNN分类器
model.fit(input_feature, dataset[1].values)    # 训练模型

# 预测新文本
test_query = "帮我导航到南京路"                  # 要预测的文本
test_sentence = " ".join(jieba.lcut(test_query))  # 分词处理
test_feature = vector.transform([test_sentence])   # 转换为数字特征

# KNN输出预测结果
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature)[0])


# 2.逻辑回归
# 使用逻辑回归模型
model = LogisticRegression(random_state=42, max_iter=1000)  # 设置参数
model.fit(input_feature, dataset[1].values)

# 逻辑回归输出预测结果
print("待预测的文本:", test_query)
print("逻辑回归预测结果:", model.predict(test_feature)[0])

# 3.决策树
# 中文分词处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 文本向量化 - 决策树通常用CountVectorizer效果更好
vector = CountVectorizer(
  max_features=1000,     # 最大特征数(只保留词频最高的1000个词作为特征)
  min_df=2,              # 文档频率下限(词必须在至少2个文档中出现才会被保留)
  max_df=0.95            # 文档频率上限(出现在超过95%文档中的词会被过滤掉)
)
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 使用决策树模型
model = DecisionTreeClassifier(
    criterion='gini',           # 分割标准
    max_depth=10,              # 最大深度
    min_samples_split=5,       # 内部节点最少样本数  
    min_samples_leaf=2,        # 叶节点最少样本数
    random_state=42           # 随机种子
)
model.fit(input_feature, dataset[1].values)

# 预测测试文本
test_query = "播放周杰伦的歌"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

# 输出结果
print("待预测的文本:", test_query)
print("决策树预测结果:", model.predict(test_feature)[0])


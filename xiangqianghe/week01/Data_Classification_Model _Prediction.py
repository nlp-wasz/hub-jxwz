import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier  #KNN模块
from sklearn import linear_model    #线性模块
from sklearn import tree    #新增决策树模块


dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset[1].value_counts())
print(dataset[1].head(5))

# 提取文本的特征tfdif， dataset[0]
# 构建一个模型knn， 学习 提取的特征和 标签dataset[1]的关系
# 预测 用户输入的一个文本，输出预测结果

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))#对每一行使用函数进行处理,使用空格进行拼接，便于sklearn对中文进行处理
print(input_sententce)

#将文本转换为词频（TF）向量，也就是统计每个词在每个句子中出现的次数，形成一个稀疏矩阵。
vector = CountVectorizer() #对文本进行特征提取，词袋模型，统计每个词出现的频率
vector.fit(input_sententce.values) #从所有文本中学习词汇表
print(vector.vocabulary_) #打印出每个词对应的编号，例如
input_feature = vector.transform(input_sententce.values) #从所有输入文本中学习一个词汇表（即都有哪些词）。将每个句子转换为对应的词频向量（稀疏矩阵形式）。

# 使用KNN模型进行预测
model = KNeighborsClassifier()  # 默认使用 5 个最近邻
model.fit(input_feature, dataset[1].values) # 用特征和标签训练模型
print(model)

test_query = "帮我导航到昆明"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("KNN模型预测结果：", model.predict(test_feature))

test_query = "帮我播放凡人修仙传"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("KNN模型预测结果：", model.predict(test_feature))

# 使用决策树模型进行预测
model = tree.DecisionTreeClassifier()   #模型初始化
model.fit(input_feature, dataset[1].values)

test_query = "帮我导航到北京"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("决策树模型预测结果：", model.predict(test_feature))

test_query = "我想看lol世界比赛"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("决策树模型预测结果：", model.predict(test_feature))

# 使用线性回归模型进行预测
model = linear_model.LogisticRegression(max_iter=1000)  #模型初始化，人工设置了超参数， 从训练集学习到的叫模型参数
model.fit(input_feature, dataset[1].values)

test_query = "我想听王菲的歌"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("线性回归模型预测结果：", model.predict(test_feature))

test_query = "我想听南山南"
test_sententce = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sententce])

print("待预测的文本", test_query)
print("线性回归模型预测结果：", model.predict(test_feature))

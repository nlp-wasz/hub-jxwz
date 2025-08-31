import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn import neighbors
from sklearn import linear_model # 线性模型模块
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset)
# X, y = dataset[0], dataset[1]
# train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=999, test_size=0.3) # 数据切分 30% 样本划分为测试集
# print("train_x", train_x.values)
# print("真实标签", test_y)

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
print("input_sententce.value", input_sententce.values)

vector = TfidfVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = tree.DecisionTreeClassifier() # 模型初始化
model.fit(input_feature, dataset[1].values)

test_query = "帮我播放一下张三的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("决策树的预测结果: ", model.predict(test_feature))

model = neighbors.KNeighborsClassifier(n_neighbors=5) # 模型初始化
model.fit(input_feature, dataset[1].values)
print("KNN-5的预测结果: ", model.predict(test_feature))

model = linear_model.LogisticRegression(max_iter=1000) # 模型初始化， 人工设置的参数叫做超参数， 模型参数可以从训练集学习到的
model.fit(input_feature, dataset[1].values)
print("逻辑回归的预测结果: ", model.predict(test_feature))

model = RandomForestClassifier()
model.fit(input_feature, dataset[1].values)
print("随机森林的预测结果: ", model.predict(test_feature))

"""
决策树
随机森林
贝叶斯
"""

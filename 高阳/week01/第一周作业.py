import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# sklearn对中文处理
input_sententce = dataset[0].apply(lambda x: ",".join(jieba.lcut(x)))
# 对文本进行提取特征 默认是使用标点符号分词
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 逻辑回归
model = LogisticRegression()
model.fit(input_feature, dataset[1].values)
# 预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = ",".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("逻辑回归模型预测结果: ", model.predict(test_feature))

# KNN模型
model = KNeighborsClassifier(n_neighbors=5)
model.fit(input_feature, dataset[1].values)
# 预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = ",".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("KNN模型预测结果: ", model.predict(test_feature))

# 决策树
model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
# 预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = ",".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("决策树模型预测结果: ", model.predict(test_feature))

# 支持向量机
model = SVC(kernel='linear')
model.fit(input_feature, dataset[1].values)
# 预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = ",".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("支持向量机模型预测结果: ", model.predict(test_feature))

# 随机森林
model = RandomForestClassifier(n_estimators=10)
model.fit(input_feature, dataset[1].values)
# 预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = ",".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("随机森林模型预测结果: ", model.predict(test_feature))





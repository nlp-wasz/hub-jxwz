
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
import numpy as np

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 划分数据集和测试集， 方便测试模型效果
train_x, test_x, train_y, test_y = train_test_split(input_feature, dataset[1].values, test_size = 0.1,random_state=520) # 数据切分 10% 样本划分为测试集


KNN_model = KNeighborsClassifier(n_neighbors = 1)
KNN_model.fit(train_x, train_y)
prediction = KNN_model.predict(test_x)
print("1-KNN的预测结果", (test_y == prediction).sum(),"/", np.shape(test_x)[0])

Linear_model = linear_model.LogisticRegression(max_iter=1000)
Linear_model.fit(train_x, train_y)
prediction = Linear_model.predict(test_x)
print("逻辑回归的预测结果", (test_y == prediction).sum(),"/", np.shape(test_x)[0])

DecisionTree_model = tree.DecisionTreeClassifier()
DecisionTree_model.fit(train_x, train_y)
prediction = DecisionTree_model.predict(test_x)
print("决策树的预测结果", (test_y == prediction).sum(),"/", np.shape(test_x)[0])



test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", KNN_model.predict(test_feature))
print("逻辑回归模型预测结果: ", Linear_model.predict(test_feature))
print("决策树模型预测结果: ", DecisionTree_model.predict(test_feature))

test_query = "立刻帮我建一个备忘录2月3号开会"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", KNN_model.predict(test_feature))
print("逻辑回归模型预测结果: ", Linear_model.predict(test_feature))
print("决策树模型预测结果: ", DecisionTree_model.predict(test_feature))

# HashingVectorizer 与 KNN模型

import jieba
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model # 线性模型模块

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
print(input_sententce)

corpus = input_sententce

vectorizer = HashingVectorizer(n_features=2**20)
X = vectorizer.transform(corpus)
print(X.shape)
input_feature = X

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)

vector = vectorizer

print("----------test1----------")
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))

print("----------test2----------")
test_query = "帮我找一个欢快的歌曲"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))

print("----------test3----------")
test_query = "帮我关一下闹钟"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
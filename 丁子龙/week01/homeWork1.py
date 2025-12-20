# 该代码使用CountVectorizer 与 线性模型


import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model # 线性模型模块


dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
print(input_sententce)

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

X = input_feature
y = dataset[1]
model = linear_model.LogisticRegression(max_iter=1200)
model.fit(X, y) # fit 就是训练模型
print(model)

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


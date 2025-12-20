from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('dataset.csv', sep='\t', header=None)

# 将每条数据中文分词生成特征向量作为训练数据
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 生成特征向量
vector = CountVectorizer()
vector.fit(input_sentence.values) # 接收文本数据
input_feature = vector.transform(input_sentence.values) #开始转换为特征向量

# 1、使用KNN模型训练数据
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

test_query = "我已经走到人民大道了"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))

# 2、
model_svc = LinearSVC(class_weight="balanced")
model_svc.fit(input_feature, dataset[1].values)

test_query = "我已经走到人民大道了"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("SVC模型预测结果: ", model_svc.predict(test_feature))


model_bayes = MultinomialNB(alpha=0.1)  # 平滑参数
model_bayes.fit(input_feature, dataset[1].values)

test_query = "我已经走到人民大道了"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("Bayes模型预测结果: ", model_bayes.predict(test_feature))

# 待预测的文本 我已经走到人民大道了
# KNN模型预测结果:  ['HomeAppliance-Control']
# 待预测的文本 我已经走到人民大道了
# SVC模型预测结果:  ['Radio-Listen']
# 待预测的文本 我已经走到人民大道了
# Bayes模型预测结果:  ['Radio-Listen']

# 这里因为使用了KNN预测结果不准确，所以问了DS有没有更准确的中文分词向量的预测模型，找了两个，但是结果依旧不准确，想问下老师，是哪里有问题





import pandas as pd
import jieba
import sklearn.feature_extraction.text
from sklearn.linear_model import LogisticRegression
from sklearn import svm

datacsv = pd.read_csv('.\dataset.csv', sep="\t", header=None)
datalcut = datacsv[0].apply(lambda m: " ".join(jieba.lcut(m)))
feature_vector = sklearn.feature_extraction.text.CountVectorizer()
feature_vector.fit(datalcut.values)
feature_values = feature_vector.transform(datalcut.values)

#训练逻辑回归模型
model_logistic = LogisticRegression(max_iter=100)
model_logistic.fit(feature_values, datacsv[1])
print("model_logistic  逻辑回归模型训练完成", end="\n")

#训练支持向量机模型
model_svm = svm.SVC(kernel='linear', C=10, gamma=0.1)
model_svm.fit(feature_values, datacsv[1])
print("model_svm  支持向量机分类模型训练完成", end="\n\n")

#测试逻辑回归模型
print("请输入需要预测的文本语句：")
predict_input = input()
predict_cut = " ".join(jieba.lcut(predict_input))
feature_predict = feature_vector.transform([predict_cut])
prediction = model_logistic.predict(feature_predict)
print(f"逻辑回归模型的预测结果：{prediction}", end="\n\n")

#测试支持向量机模型
prediction = model_svm.predict(feature_predict)
print(f"支持向量机模型的预测结果：{prediction}", end="\n\n")

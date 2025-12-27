import pandas as pd
import jieba
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier;



# 读取文件中的数据
datas = pd.read_csv('dataset.csv', sep='\t', header=None, names=['text', 'label'])
#逻辑回归训练数据
logistic_regression_model = linear_model.LogisticRegression()
#分词处理
words = datas['text'].apply(lambda x: " ".join(jieba.lcut(x)))
print(words)
vectorizer = CountVectorizer()
vectorizer.fit(words)
train_data = vectorizer.transform(words)

逻辑回归
regression_model = linear_model.LogisticRegression(max_iter=1000)
regression_model.fit(train_data, datas['label'])
input_data = input("请输入你要用逻辑回归预测的字段：")
cut_test_data = " ".join(jieba.lcut(input_data))
test_data = vectorizer.transform([cut_test_data])
print(regression_model.predict(test_data))

#KNN模型
knn_model = KNeighborsClassifier()
knn_model.fit(train_data, datas['label'])
knn_input_data = input("请输入你要用KNN模型预测的字段：")
knn_cut_test_data = " ".join(jieba.lcut(knn_input_data))
knn_test_data = vectorizer.transform([knn_cut_test_data])
print(knn_model.predict(knn_test_data))
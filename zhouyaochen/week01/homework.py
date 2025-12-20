import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

dataset=pd.read_csv("dataset.csv",sep="\t",header=None)
print(dataset.head(10))


'''
1.KNN模型完成文本分类操作
'''
dataset_apply= dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(dataset_apply.values)
feature = vector.transform(dataset_apply.values)

model = KNeighborsClassifier()
model.fit(feature, dataset[1].values)
print(model)

test_query = "这个周末的天气如何"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))

'''
2.决策树模型完成文本分类操作
'''
model = tree.DecisionTreeClassifier()
model.fit(feature, dataset[1].values)
print(model)

test_query="小度小度 打开空调"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("决策树模型预测结果: ", model.predict(test_feature))


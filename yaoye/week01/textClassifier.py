import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
#print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
#print(input_sententce)

train_x, test_x0, train_y, test_y = train_test_split(input_sententce.values, dataset[1].values, random_state=666) # 数据切分 25% 样本划分为测试集

vector = CountVectorizer()
vector.fit(train_x)
input_feature = vector.transform(train_x)
test_x = vector.transform(test_x0)

model = KNeighborsClassifier()
model.fit(input_feature, train_y)
# print(model)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
prediction = model.predict(test_x)
print("测试集预测结果", prediction)
print("knn rate: ", (test_y == prediction).sum(), len(test_x0))

model = linear_model.LogisticRegression(max_iter=1000)
model.fit(input_feature, train_y)
prediction = model.predict(test_feature)
print("逻辑回归的预测结果", prediction)
prediction = model.predict(test_x)
print("测试集预测结果", prediction)
print("logistic regression rate: ", (test_y == prediction).sum(), len(test_x0))
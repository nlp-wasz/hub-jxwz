import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier #导入 knn 模型
from sklearn.tree import DecisionTreeClassifier #决策树模型
from sklearn.linear_model import LogisticRegression #线性模型
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) #sklearn对中文处理
# print(input_sentence)

vector = CountVectorizer() #对文本提取特征
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

#训练模型 KNN
# model = KNeighborsClassifier()
# model.fit(input_feature, dataset[1].values)
# print(model)

#决策树模型 tree
# model = DecisionTreeClassifier()
# model.fit(input_feature, dataset[1].values)
# print(model)

#线性模型 line
model = LogisticRegression()
model.fit(input_feature, dataset[1].values)
print(model)


#用户提问
test_query = "帮我播放十个勤天的歌"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测文本", test_query)
# print("KNN模型预测结果", model.predict(test_feature))
# print("决策树模型预测结果", model.predict(test_feature))
print("线性模型预测结果", model.predict(test_feature))

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model #逻辑回归
from sklearn import tree #决策树


dataset = pd.read_csv('../dataset.csv',sep="\t",header=None)
# print(dataset[0])


#提取文本的特征 tfidf, dataset[0]
#构建一个模型 knn,学习提取的特征 和 标签dataset[1] 的关系
#预测，用户输入的一个文本，进行结果预测

input_sentence = dataset[0].apply(lambda x:" ".join(jieba.lcut(x))) #对每一行使用函数进行处理
# print(input_sentence.values)

vector = CountVectorizer() #对文本进行提取特征 默认使用标点符号分词
temp = vector.fit(input_sentence.values)
# print(vector.vocabulary_)
input_feature = vector.transform(input_sentence.values)
# print(input_feature)


model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
# print(dataset[1].values)

model1 = linear_model.LogisticRegression(max_iter=1000)
model1.fit(input_feature, dataset[1].values)


model2 = tree.DecisionTreeClassifier()
model2.fit(input_feature, dataset[1].values)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

print("待预测的文本", test_query)
print("KNN模型预测结果：",model.predict(test_feature))
print("逻辑回归模型预测结果：",model1.predict(test_feature))
print("决策树模型预测结果：",model2.predict(test_feature))
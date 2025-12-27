import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
# # 加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# #jieba分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 提取特征
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 测试数据  提取特征
test_query = "现在有金星脱口秀"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

# 训练KNN模型
model = KNeighborsClassifier(n_neighbors=5)
model.fit(input_feature, dataset[1].values)

print("待预测的文本", test_query)
print("KNN模型预测", model.predict(test_feature))

# 训练决策树模型
model = tree.DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
print("决策树模型预测", model.predict(test_feature))

# 朴素贝叶斯模型
model = MultinomialNB()
model.fit(input_feature, dataset[1].values)
print("待预测的文本", test_query)
print("朴素贝叶斯模型预测", model.predict(test_feature))

#SVN模型
model = SVC(kernel="linear")
model.fit(input_feature, dataset[1].values)
print("待预测的文本", test_query)
print("SVN模型预测", model.predict(test_feature))



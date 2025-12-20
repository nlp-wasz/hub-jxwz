import pandas as pd  # 结构化数据
import jieba  # 中文分词
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征提取
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯模型

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)  # 读取并结构化数据

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 中文分词

# 对文本进行提取特征
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 实例化并训练模型
# KNN模型
knn_model = KNeighborsClassifier()
knn_model.fit(input_feature, dataset[1].values)
# 朴素贝叶斯模型
nb_model = MultinomialNB()
nb_model.fit(input_feature, dataset[1].values)

# 设定预测文本并提取特征
test_query = "帮我导航到北京"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

# 输出预测结果
print("\n待预测的文本：", test_query)
print("KNN模型预测结果：", knn_model.predict(test_feature)[0])
print("朴素贝叶斯预测结果：", nb_model.predict(test_feature)[0])

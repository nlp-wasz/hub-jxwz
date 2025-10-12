import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.cut(x))) # sklearn对中文处理
print(input_sentence)

vector = TfidfVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sentence.values)
input_feature1 = input_feature2 = vector.transform(input_sentence.values)
print(input_feature1)

#KNN
model1 = KNeighborsClassifier()
model1.fit(input_feature1, dataset[1].values)


#决策树
model2 = tree.DecisionTreeClassifier()
model2.fit(input_feature2, dataset[1].values)


test_query = "鸣潮这个游戏太好玩了"
test_sentence = " ".join(jieba.cut(test_query))
test_feature = vector.transform([test_sentence])
print(f"待预测的文本: {test_query} \n"
      f"KNN模型预测结果: :{model1.predict(test_feature)} \n"
      f"决策树模型预测结果::{model2.predict(test_feature)} \n")

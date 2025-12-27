import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 tree， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理


vectorizer = TfidfVectorizer()  # 必须先初始化

input_feature = vectorizer.fit_transform(input_sententce.values)#fit()：用于学习数据特征，建立转换规则，transform():根据学到的规则转换数据


model_2 = DecisionTreeClassifier()
model_2.fit(input_feature, dataset[1].values)
print(model_2)

test_query = "去酒店的路怎么走"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vectorizer.transform([test_sentence])
print("待预测的文本", test_query)
print("DecisionTree模型预测结果: ", model_2.predict(test_feature))



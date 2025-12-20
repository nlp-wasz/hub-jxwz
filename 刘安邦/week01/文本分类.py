"""
使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# 模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树模块

# 加载数据集
dataset = pd.read_csv('dataset.csv', sep='\t', header=None)

# 处理数据
input_sentence = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))

# 提取特征
vector = CountVectorizer()  # sklearn
# 构建词表
vector.fit(input_sentence.values)  # 从input_sentence提取词表。输入是字符串列表，不修改输入
# print(vector.vocabulary_)  # 查看词表
# 转换为特征向量
input_vector = vector.transform(input_sentence.values)  # 根据提取的词表，把每句转换为特征向量
# print(input_vector)

# 模型1：knn
# 加载模型
model_knn = KNeighborsClassifier()
# 训练模型
model_knn.fit(input_vector, dataset[1])  # 特征向量, 标签

# 模型2：决策树
# 加载模型
model_tree = DecisionTreeClassifier()
# 训练模型
model_tree.fit(input_vector, dataset[1])  # 特征向量, 标签

# 测试模型
# 重复数据处理
test_query = ['去二仙桥怎么走', '我想吃火锅', '你那里下冰雹了吗']
print('输入文本：', test_query)
test_sentence = [' '.join(jieba.lcut(query)) for query in test_query]
print(test_sentence)
test_vector = vector.transform(test_sentence)
# 预测
prediction_knn = model_knn.predict(test_vector)
print('knn预测结果', prediction_knn)
prediction_tree = model_tree.predict(test_vector)
print('决策树预测结果', prediction_tree)

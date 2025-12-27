import jieba
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn import neighbors

# from sklearn.

# 添加“header=None”，告诉函数，我们读取的原始文件数据没有列索引。因此，read_csv为自动加上列索引。
dataset = pd.read_csv('dataset.csv',sep='\t',header=None)
# print(dataset[1].value_counts())

# 将文本的第一列使用jieba进行分词
# 用空格将所有分词进行拼接
input_sentence = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))
# print(input_sentence)

# 对所有分词进行特征向量提取
vector = CountVectorizer()
# fit训练以后，向vector中增加分词和的数据转换映射
vector.fit(input_sentence)
# print(vector.vocabulary_)
# 转换特征向量
input_feature = vector.transform(input_sentence)

# 逻辑回归模型
module = linear_model.LogisticRegression()
module.fit(input_feature, dataset[1])
print(module)

test_query = "帮我导航到北京"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
prediction = module.predict(test_feature)
print("测试问题：",test_query)
print("逻辑回归预测结果：",prediction)


# 决策树
module = tree.DecisionTreeClassifier()
module.fit(input_feature, dataset[1])
print(module)

test_query = "王力宏演唱会是什么时候"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
prediction = module.predict(test_feature)
print("测试问题：",test_query)
print("决策树预测结果：",prediction)

# KNN
module = neighbors.KNeighborsClassifier(n_neighbors=5)
module.fit(input_feature, dataset[1])
print(module)

test_query = "周一天气如何"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
prediction = module.predict(test_feature)
print("测试问题：",test_query)
print("KNN-5预测结果：",prediction)

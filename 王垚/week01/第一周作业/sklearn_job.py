import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
from sklearn import tree
from sklearn import linear_model  # 线性模型模块
from sklearn.model_selection import train_test_split  # 数据集划分

# 1、提取文本
dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
print(dataset.head(5))

# 2、进行分词
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 3、初始化特征提取
vector = CountVectorizer()

# 4、根据分词后的文本学习词汇表
vector.fit(input_sententce.values)

# 5、将文本转换为特征向量
input_feature = vector.transform(input_sententce.values)


# 6、将数据集进行切分 方便测试 dataset[1] 代表是提取第2列的标签列
# train_feature：训练集特征
# test_feature：测试集特征
# train_label：训练集标签
# test_label：测试集标签
train_feature, test_feature, train_label, test_label = train_test_split(
    input_feature, dataset[1].values, test_size=0.3, random_state=520
)

# 7、拿部分切成的数据进行训练和测试

# 8、确定要进行预测的数据
my_test_query = "还有双鸭山到淮阴的汽车票吗13号的"
my_test_sentence = " ".join(jieba.lcut(my_test_query))
my_test_feature = vector.transform([my_test_sentence])

# 9、加载模型
# 10、训练模型
# 11、测试准确率

# 加载逻辑回归模型
model = linear_model.LogisticRegression(max_iter=1000)
# 训练逻辑回归模型
model.fit(train_feature, train_label)
prediction = model.predict(test_feature)
print("预测结果\n", prediction)
accuracy = (prediction == test_label).sum() / len(test_label)
print(f"逻辑回归模型准确率：{accuracy:.4f}")
print("待预测的文本", my_test_query)
print("逻辑回归模型预测结果: ", model.predict(my_test_feature))


# 加载决策树模型
model = tree.DecisionTreeClassifier()
# 训练逻辑回归模型
model.fit(train_feature, train_label)
prediction = model.predict(test_feature)
print("预测结果\n", prediction)
accuracy = (prediction == test_label).sum() / len(test_label)
print(f"决策树模型准确率：{accuracy:.4f}")
print("待预测的文本", my_test_query)
print("决策树模型预测结果: ", model.predict(my_test_feature))


# 加载KNN模型
model = neighbors.KNeighborsClassifier()
# 训练逻辑回归模型
model.fit(train_feature, train_label)
prediction = model.predict(test_feature)
print("预测结果\n", prediction)
accuracy = (prediction == test_label).sum() / len(test_label)
print(f"KNN模型准确率：{accuracy:.4f}")
print("待预测的文本", my_test_query)
print("KNN模型预测结果: ", model.predict(my_test_feature))


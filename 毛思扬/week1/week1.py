# 使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。（注意：这个作业代码实操提交）
import pandas  # 数据预处理清洗啥的
import jieba  # 分词
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import classification_report  # F1分数

from sklearn.feature_extraction.text import CountVectorizer  # 特征提取方法 词袋模型
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取方法 TF-IDF

from sklearn.neighbors import KNeighborsClassifier  # 预测模型 最近邻
from sklearn.linear_model import LogisticRegression  # 预测模型 逻辑回归
from sklearn.svm import SVC  # 预测模型 支持向量机（SVM）
from sklearn.ensemble import RandomForestClassifier  # 预测模型 随机森林
from sklearn.naive_bayes import MultinomialNB  # 预测模型 朴素贝叶斯


# 参考资料https://www.51cto.com/article/804256.html https://blog.csdn.net/weixin_42608414/article/details/88046380
# 1.数据集读取
data = pandas.read_csv("dataset.csv", sep="\t", header=None)

# 2.分词处理
input_text = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
print(type(input_text))
print(type(input_text.values))
print(input_text.values)

# 3.特征提取-词袋模型(词袋模型是一种简单的文本表示方法，它将文本转换为词频向量)
bow = CountVectorizer()  # 词袋模型 bag of words
bow_feature = bow.fit_transform(input_text.values)
print(type(bow_feature))
print(f"词袋模型矩阵维度：{bow_feature.shape}")
# 获取特征名称
feature_names = bow.get_feature_names_out()
# 打印词频矩阵
print(f"词袋模型特征名称:{feature_names}")
print("词袋模型词频矩阵:\n", bow_feature.toarray())

# 3.特征提取-TF-IDF 向量化(TF-IDF（Term Frequency-Inverse Document Frequency）
# 是一种更高级的文本表示方法，它不仅考虑词频，还考虑了词的重要性)
tfIdf = TfidfVectorizer()
tfIdf_feature = tfIdf.fit_transform(input_text.values)
print(type(tfIdf_feature))
print(f"TF-IDF矩阵维度：{tfIdf_feature.shape}")
print(f"TF-IDF特征名称:{tfIdf.get_feature_names_out()}")
print(f"TF-IDF词频矩阵:\n{tfIdf_feature.toarray()}")


# 4.训练 and 结果输出
def train_and_predict(model, features, labels, model_name):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25,
                                                        random_state=42)
    # 加载并训练模型
    model.fit(X_train, y_train)
    # 预测并输出结果
    prediction = model.predict(X_test)
    print(f"{model_name}预测结果", (y_test == prediction).sum(), X_test.shape[0])
    print(f"{model_name}准确率", (y_test == prediction).sum() / X_test.shape[0])
    print(classification_report(y_test, prediction))


# 5.KNN 分类
KNNModel = KNeighborsClassifier()
# 词袋模型训练与预测
train_and_predict(KNNModel, bow_feature, data[1], "KNN-BOW")
# TF-IDF模型训练与预测
train_and_predict(KNNModel, tfIdf_feature, data[1], "KNN-TFIDF")

# 5.逻辑回归分类
logisticRegressionModel = LogisticRegression()
train_and_predict(logisticRegressionModel, bow_feature, data[1], "logisticRegression-BOW")
train_and_predict(logisticRegressionModel, tfIdf_feature, data[1], "logisticRegression-TFIDF")

# 5.支持向量机SVM分类
svmModel = SVC()
train_and_predict(svmModel, bow_feature, data[1], "SVM-BOW")
train_and_predict(svmModel, tfIdf_feature, data[1], "SVM-TFIDF")

# 5.随机森林分类
randomForestModel = RandomForestClassifier()
train_and_predict(randomForestModel, bow_feature, data[1], "RandomForest-BOW")
train_and_predict(randomForestModel, tfIdf_feature, data[1], "RandomForest-TFIDF")

# 5.朴素贝叶斯分类
naiveBayesModel = MultinomialNB()
train_and_predict(naiveBayesModel, bow_feature, data[1], "NaiveBayes-BOW")
train_and_predict(naiveBayesModel, tfIdf_feature, data[1], "NaiveBayes-TFIDF")

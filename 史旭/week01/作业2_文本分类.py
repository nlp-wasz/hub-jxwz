import pandas as pd
import jieba as jb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn import neighbors


def getPredictStr(vector: CountVectorizer, predict_str: str):
    # 预测数据 分词处理和提取特征向量
    predict_str = " ".join(jb.lcut(predict_str))
    predict_X = vector.transform([predict_str])

    return predict_X


# 1.读取 csv文件 获取数据
data = pd.read_csv("dataset.csv", header=None, sep="\t")

# 2.获取 数据和标签
X_data = data[0]
y = data[1]

# 3.通过jieba中文分词工具 进行分词处理
X_data = X_data.apply(lambda d: " ".join(jb.lcut(d)))

# 4.提取特征向量
vector = CountVectorizer()
# 根据分词后的文本数据 构建词频表
vector.fit(X_data)
# 训练获取词频向量
X = vector.transform(X_data)

# 5.模型训练
# 获取预测数据 特征向量
predict = "明天是周末！"
predict_X = getPredictStr(vector, predict)

# ①LogisticRegressionCV
lrCV_model = linear_model.LogisticRegressionCV()
lrCV_model.fit(X, y)
lrCV_predict = lrCV_model.predict(predict_X)
print(f"LogisticRegressionCV--预测结果：{lrCV_predict}")

# ②LinearSvc
lsvc_odel = LinearSVC()
sc = lsvc_odel.fit(X, y)
lsvc_predict = lsvc_odel.predict(predict_X)
print(f"LinearSvc--预测结果：{lsvc_predict}")

# 逻辑回归，决策树，KNN
lr_model = linear_model.LogisticRegression()
lr_model.fit(X, y)
lr_predict = lr_model.predict(predict_X)
print(f"LogisticRegression--预测结果：{lr_predict}")


dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X, y)
dt_predict = dt_model.predict(predict_X)
print(f"DecisionTreeClassifier--预测结果：{dt_predict}")

knn_model = neighbors.KNeighborsClassifier()
knn_model.fit(X, y)
knn_predict = knn_model.predict(predict_X)
print(f"KNeighborsClassifier--预测结果：{knn_predict}")
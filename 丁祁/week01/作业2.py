import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# 2、使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。（注意：这个作业代码实操提交）

def test_knn_model(test_list):
    # KNN 分类器，默认用最近的 5 个邻居投票
    for k in [1, 3, 5, 7, 9]:
        model = KNeighborsClassifier(n_neighbors=k)
        # 把“数字矩阵”和“正确答案（第 1 列标签）”喂给 KNN，让它开始学习
        model.fit(input_feature, dataset[1].values)
        print(f"KNN model: {model}")
        model_predict_test_qurey(model, test_list)


def test_logistic_model(test_list):
    # 逻辑回归实现
    model_l = LogisticRegression(max_iter=1000)
    model_l.fit(input_feature, dataset[1].values)
    model_predict_test_qurey(model_l, test_list)


def test_tree_model(test_list):
    # 决策树实现
    model_tree = DecisionTreeClassifier()
    model_tree.fit(input_feature, dataset[1].values)
    model_predict_test_qurey(model_tree, test_list)


def model_predict_test_qurey(model, test_list):
    print("\n" + "---" * 10)
    for test_query in test_list:
        test_sentenct = " ".join(jieba.lcut(test_query))
        test_feature = vector.transform([test_sentenct])
        print(f"待预测文本：{test_query}")
        print(f"{model}----模型预测结果：{model.predict(test_feature)}")


if __name__ == '__main__':
    dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
    print(dataset.head(3))

    # dataset 第一列数据进行分词，并用空格将分词结果拼接
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
    # 对文本进行特征提取，默认是用标点符号分词
    vector = CountVectorizer()
    # 学习一下都有哪些词
    vector.fit(input_sententce.values)
    # 把整列句子真正变成数字矩阵（一行一个样本，一列一个词）
    input_feature = vector.transform(input_sententce.values)
    test_list = ["明天天气如何？", "青藏高原有多高？", "苹果和李子哪个是红的？"]

    # knn 模型实现
    test_knn_model(test_list)

    # 逻辑回归实现
    test_logistic_model(test_list)

    # 决策树实现
    test_tree_model(test_list)






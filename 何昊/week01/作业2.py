import jieba
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

filename = "dataset.csv"


# KNN 模型训练
def knn_train():
    # 读取数据集
    dataset = pd.read_csv(filename, sep="\t", header=None)
    # 对数据集进行预处理, 分词再使用空格分隔, sklearn 不能处理中文, 需要人工分词, 空格充当标点符号
    input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
    # 构建词频矩阵
    vector = CountVectorizer()
    vector.fit(input_sentence.values)
    # 使用 KNN 模型
    model = KNeighborsClassifier(n_neighbors=3)
    # 转换特征向量, 并使用 KNN 存储训练集, 与其他模型不同, 不学习任何参数
    model.fit(vector.transform(input_sentence.values), dataset[1].values)
    print("\n-------------------KNN模型训练完成-----------------\n")
    # 预测测试样本
    test_query = "这是一个测试样本"
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print("待预测的文本: ", test_query)
    print("KNN模型预测结果: ", model.predict(test_feature))


# 逻辑回归模型训练
def lr_train():
    # 读取数据集
    dataset = pd.read_csv(filename, sep="\t", header=None, names=['sentence', 'label'])
    # 分词处理
    input_sentence = dataset['sentence'].apply(lambda x: " ".join(jieba.lcut(x)))
    # 构建词频矩阵
    vector = CountVectorizer()
    x = vector.fit_transform(input_sentence.values)
    y = dataset['label']
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 使用逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    # 转换特征向量, 并学习参数
    model.fit(x_train, y_train)
    print("\n-------------------逻辑回归模型训练完成-----------------\n")
    # 预测测试集
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("模型准确率: ", accuracy)
    print("模型精确率: ", precision)
    print("模型召回率: ", recall)
    print("模型F1值:  ", f1)


def main():
    knn_train()
    lr_train()


if __name__ == '__main__':
    main()


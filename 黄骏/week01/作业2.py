import  pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 导入数据集
dataset = pd.read_csv("dataset.csv", sep = "\t", header=None)
# 数据集划分
X = dataset[0]
y = dataset[1]
tran_x, text_x, tran_y, text_y = train_test_split(X, y, random_state = 666)
# 训练数据处理
tran_sentence = tran_x.apply(lambda x: " ".join(jieba.lcut(x)))
# 训练数据特征提取
vectorizer = CountVectorizer()
vectorizer.fit(tran_sentence.values)
tran_feature = vectorizer.transform(tran_sentence.values)
# 训练模型
model1 = KNeighborsClassifier() # KNN模型
model1.fit(tran_feature, tran_y.values)
model2 = LogisticRegression()   # 逻辑回归模型
model2.fit(tran_feature, tran_y.values)
# 测试数据处理
test_sentence = text_x.apply(lambda x: " ".join(jieba.lcut(x)))
# 测试数据特征提取
test_feature = vectorizer.transform(test_sentence.values)
# 模型推理
prediction1 = model1.predict(test_feature)
prediction2 = model2.predict(test_feature)
# 预测结果
print(f"KNN模型的预测结果：{(prediction1 == text_y).sum()}/{len(text_y)}")
print(f"KNN模型的预测结果：{(prediction2 == text_y).sum()}/{len(text_y)}")

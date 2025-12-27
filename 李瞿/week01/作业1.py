import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../data/dataset.csv', sep='\t', header=None)
print(data.head())
X, y = data[0], data[1]

# 1.提取文本的特征 CountVectorizer
sentence = X.apply(lambda x: " ".join(jieba.lcut(x)))  # 使用lambda函数
vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(sentence)

# 2.模型列表
models = [KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(), SVC(), RandomForestClassifier()]
for model in models:
    model.fit(vec, y)
    # 3.评估
    query = "导航到北京"
    sentence = [" ".join(jieba.lcut(query))]
    input_features = vectorizer.transform(sentence)
    print(f'模型选择:{model}, 预测结果为:{model.predict(input_features)}')

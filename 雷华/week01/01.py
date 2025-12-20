import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
df = pd.read_csv('dataset.csv')
texts = df['text'].fillna('').values  # 填充缺失值
labels = df['label'].values

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 特征提取 (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. 模型1: 逻辑回归
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

# 5. 模型2: 朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

# 6. 结果评估
print("===== 模型性能对比 =====")
print(f"逻辑回归准确率: {accuracy_score(y_test, lr_pred):.4f}")
print(f"朴素贝叶斯准确率: {accuracy_score(y_test, nb_pred):.4f}\n")

print("===== 逻辑回归分类报告 =====")
print(classification_report(y_test, lr_pred))

print("===== 朴素贝叶斯分类报告 =====")
print(classification_report(y_test, nb_pred))

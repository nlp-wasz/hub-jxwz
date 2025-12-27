# homework2.py
# -*- coding: utf-8 -*-
"""
Function: This script performs text classification using SVM and XGBoost on a dataset.
Created on Wed Aug 13 2025
v0.1 baseline
    - SVM accuracy 89.13%; XGBoost accuracy 83.10%
v0.2 
    1. working directory set to script location
    2. Increased TF-IDF features to 5000
    3. Added Label Encoding for XGBoost
    4. Tuned XGBoost hyperparameters
    -SVM accuracy 89.46%; XGBoost accuracy 84.58%
"""
import os
import chardet
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 确保当前工作目录正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"当前工作目录: {os.getcwd()}")

# 1. 数据加载和预处理（保持不变）
with open('dataset.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
print("检测到的编码:", result['encoding'])

df = pd.read_csv('dataset.csv', 
                 header=None, 
                 names=['raw_data'],
                 encoding=result['encoding'])

df[['text', 'label']] = df['raw_data'].str.split('\t', expand=True)
df = df.drop(columns=['raw_data']).dropna()
df['label'].value_counts()

# 2. 中文分词（保持不变）
jieba.add_word("双鸭山")
jieba.add_word("墓王之王")
texts = [' '.join(jieba.cut(str(text))) for text in df['text']]
labels = df['label'].values

# 3. TF-IDF向量化（调整特征数）
tfidf = TfidfVectorizer(
    max_features=5000,  # 特征数
    ngram_range=(1, 2), # 使用1-gram和2-gram
    min_df=3,
    max_df=0.8
)
X_tfidf = tfidf.fit_transform(texts)

# 4. 划分数据集
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_tfidf, labels, df.index, test_size=0.2, random_state=42
)

# 标签编码（XGBoost需要数值型标签）
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 5. 训练SVM（保持不变）
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSVM分类报告:")
print(classification_report(y_test, y_pred_svm))

# 6. 训练优化后的XGBoost模型
xgb = XGBClassifier(
    learning_rate=0.05,      # 更小的学习率
    n_estimators=300,       # 树的数量
    max_depth=6,            # 限制树深度
    gamma=0.1,              # 分裂最小增益
    subsample=0.7,          # 数据采样比例
    colsample_bytree=0.7,   # 特征采样比例
    reg_alpha=0.1,          # L1正则化
    reg_lambda=1.0,         # L2正则化
    objective='multi:softmax',
    eval_metric='mlogloss', # 多分类指标
    random_state=42,
    tree_method='hist'     
)

# 训练模型
xgb.fit(X_train,y_train_encoded)
y_pred_xgb = le.inverse_transform(xgb.predict(X_test))

print("\nXGBoost分类报告:")
print(classification_report(y_test, y_pred_xgb))

# 7. 对比模型准确率
print(f"SVM测试集准确率: {accuracy_score(y_test, y_pred_svm):.4%}")
print(f"XGBoost测试集准确率: {accuracy_score(y_test, y_pred_xgb):.4%}")

# 8. 错误样本分析
wrong_cases_svm = df.loc[idx_test[y_test != y_pred_svm]][['text', 'label']]
print("\nSVM错误样本示例（10条）:")
print(wrong_cases_svm.sample(10))
wrong_cases_xg = df.loc[idx_test[y_test != y_pred_xgb]][['text', 'label']]
print("\nXGBoost错误样本示例（10条）:")
print(wrong_cases_xg.sample(10))
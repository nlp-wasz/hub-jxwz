# -*- coding: utf-8 -*-
"""
Function: This script performs text classification using various vectorization methods and classifiers on a dataset.
Created on Wed Aug 16 2025
v0.1 baseline
    - comment out MLPClassifier due to resources limits
"""
import os
import chardet
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer, 
                                          HashingVectorizer)
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from time import time

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"当前工作目录: {os.getcwd()}")

# 1. 数据加载和预处理
with open('dataset.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
print("检测到的编码:", result['encoding'])

df = pd.read_csv('dataset.csv', 
                 header=None, 
                 names=['raw_data'],
                 encoding=result['encoding'])

df[['text', 'label']] = df['raw_data'].str.split('\t', expand=True)
df = df.drop(columns=['raw_data']).dropna()
print("\n标签分布:")
print(df['label'].value_counts())

# 中文分词函数
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 标签编码
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# 2. 定义向量化方法
vectorizers = {
    'CountVectorizer': CountVectorizer(tokenizer=chinese_tokenizer),
    'TfidfVectorizer': TfidfVectorizer(tokenizer=chinese_tokenizer),
    'HashingVectorizer': HashingVectorizer(n_features=2**18, tokenizer=chinese_tokenizer),
    'DictVectorizer': Pipeline([
        ('count', CountVectorizer(tokenizer=chinese_tokenizer)),
        ('to_dict', lambda x: [dict(zip(range(x.shape[1]), x[i].toarray()[0])) for i in range(x.shape[0])]),
        ('dict_vec', DictVectorizer())
    ]),
    'FeatureHasher': Pipeline([
        ('count', CountVectorizer(tokenizer=chinese_tokenizer)),
        ('to_dict', lambda x: [dict(zip(range(x.shape[1]), x[i].toarray()[0])) for i in range(x.shape[0])]),
        ('hasher', FeatureHasher(n_features=2**16, input_type='dict'))
    ])
}

# 3. 定义分类模型
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', C=1, random_state=42),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'OneVsRest': OneVsRestClassifier(SVC(kernel='linear', C=1, random_state=42))
}

# 4. 训练和评估
results = []
X = df['text'].values
y = df['label_encoded'].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for vec_name, vectorizer in vectorizers.items():
    print(f"\n=== 使用 {vec_name} 向量化 ===")
    
    try:
        # 向量化处理
        start_time = time()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        vec_time = time() - start_time
        
        for model_name, model in models.items():
            print(f"训练 {model_name}...")
            start_time = time()
            model.fit(X_train_vec, y_train)
            train_time = time() - start_time
            
            start_time = time()
            y_pred = model.predict(X_test_vec)
            pred_time = time() - start_time
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            weighted_f1 = report['weighted avg']['f1-score']
            
            results.append({
                'Vectorizer': vec_name,
                'Model': model_name,
                'Accuracy': accuracy,
                'Weighted F1': weighted_f1,
                'Vectorization Time (s)': vec_time,
                'Training Time (s)': train_time,
                'Prediction Time (s)': pred_time
            })
    except Exception as e:
        print(f"{vec_name} 处理失败: {str(e)}")
        continue

# 5. 结果展示
results_df = pd.DataFrame(results)
print("\n=== 最终结果 ===")
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# 保存结果
results_df.to_csv('classification_results.csv', index=False)
print("\n结果已保存到 classification_results.csv")
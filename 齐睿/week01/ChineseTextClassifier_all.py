import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    FeatureHasher,
    HashingVectorizer
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset(file_path, header=None):
    dataset = pd.read_csv(file_path, sep="\t", header=header)
    print(f"Loaded dataset shape: {dataset.shape}")
    print(dataset.head(5))
    return dataset


def pre_process(dataset, column_index=0):
    # 添加特殊字符处理
    input_sentence = dataset[column_index].apply(
        lambda x: " ".join(j for j in jieba.lcut(x) if j.strip() and j not in ['，', '。', '！'])
    )
    return input_sentence
def pre_process_for_dict_vectorizer(dataset, column_index=0):
    """为DictVectorizer和FeatureHasher准备的预处理"""
    # 创建字典格式的特征：{词: 词频}
    dict_data = []
    for text in dataset[column_index]:
        # 分词并统计词频
        words = jieba.lcut(text)
        word_counts = {}
        for word in words:
            # 过滤标点符号
            if word.strip() and word not in ['，', '。', '！']:
                word_counts[word] = word_counts.get(word, 0) + 1
        dict_data.append(word_counts)
    return dict_data

def vectorize(input_sentence, vectorizer_type, max_features=5000):
    if vectorizer_type == 'CountVectorizer':
        vectorizer = CountVectorizer(max_features=max_features)
        input_feature = vectorizer.fit_transform(input_sentence.values)

    elif vectorizer_type == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer(max_features=max_features)
        input_feature = vectorizer.fit_transform(input_sentence.values)

    elif vectorizer_type == 'HashingVectorizer':
        vectorizer = HashingVectorizer(n_features=max_features)
        input_feature = vectorizer.fit_transform(input_sentence.values)

    elif vectorizer_type == 'DictVectorizer':
        vectorizer = DictVectorizer(sparse=True)
        input_feature = vectorizer.fit_transform(input_sentence)

    elif vectorizer_type == 'FeatureHasher':
        vectorizer = FeatureHasher(n_features=max_features)
        input_feature = vectorizer.transform(input_sentence)

    else:
        raise ValueError(f"Unsupported vectorizer: {vectorizer_type}")

    print(f"Vectorizer: {vectorizer_type}, Feature shape: {input_feature.shape}")
    return input_feature, vectorizer


def train_model(input_feature, targets_label, model_type):
    if model_type == 'KNeighbors':
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_type == 'SVM':
        model = SVC(kernel='linear', probability=True)

    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=5)

    elif model_type == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)

    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100)

    elif model_type == 'OneVsRest':
        base_model = SVC(kernel='linear', probability=True)
        model = OneVsRestClassifier(base_model)

    else:
        raise ValueError(f"Unsupported model: {model_type}")

    model.fit(input_feature, targets_label)
    print(f"Trained model: {model_type}")
    return model


def predict_model(input_feature, model):
    """
    :param input_feature: 向量化后的输入数据
    :param model: 训练的模型
    :return:
    """
    prediction = model.predict(input_feature)
    probabilities = model.predict_proba(input_feature)
    max_probability = np.max(probabilities, axis=1)

    print("Sample predictions:", prediction[:5])
    print("Sample max probabilities:", max_probability[:5])
    return prediction, probabilities, max_probability


# 主流程封装
def run_pipeline(train_file, test_file, vectorizer_type, model_type):
    # 训练流程
    train_dataset = load_dataset(train_file)
    if vectorizer_type in ['DictVectorizer', 'FeatureHasher']:
        train_sentence = pre_process_for_dict_vectorizer(train_dataset)
    else:
        train_sentence = pre_process(train_dataset)
    train_feature, vectorizer = vectorize(train_sentence, vectorizer_type)
    model = train_model(train_feature, train_dataset[1].values, model_type)

    # 测试流程
    test_dataset = load_dataset(test_file)
    if vectorizer_type in ['DictVectorizer', 'FeatureHasher']:
        test_sentence = pre_process_for_dict_vectorizer(test_dataset)
        test_feature = vectorizer.transform(test_sentence)
    else:
        test_sentence = pre_process(test_dataset)
        test_feature = vectorizer.transform(test_sentence.values)
    return predict_model(test_feature, model)


if __name__ == '__main__':
    train_data_file = "dataset.csv"
    test_data_file = "test_data.csv"
    # 尝试不同组合
    for vec in ['CountVectorizer', 'TfidfVectorizer', 'HashingVectorizer', 'DictVectorizer', 'FeatureHasher']:
        for mod in ['KNeighbors', 'SVM', 'DecisionTree', 'MLP', 'RandomForest', 'OneVsRest']:
            print(f"\n=== Running: {vec} + {mod} ===")
            run_pipeline(train_data_file, test_data_file, vec, mod)
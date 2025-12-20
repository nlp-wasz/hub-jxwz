import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC

# 1. 加载并准备数据
# 假设数据集格式为：文本\t标签
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, names=["text", "label"])
print("数据集前5行:")
print(dataset.head(5))


# 添加自定义词典（可选）
# jieba.load_userdict("custom_dict.txt")

# 2. 数据预处理 - 分词
def chinese_tokenizer(text):
    """自定义中文分词函数"""
    return jieba.lcut(text)


# 使用jieba进行分词处理
dataset["tokenized"] = dataset["text"].apply(lambda x: " ".join(jieba.lcut(x)))

# 3. 划分训练集和测试集
X = dataset["tokenized"]
y = dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4. 方法一：朴素贝叶斯分类器
print("\n===== 朴素贝叶斯分类器 =====")
# 创建管道：TF-IDF向量化 + 朴素贝叶斯   CountVectorizer/TfidfVectorizer都是文本向量化方法
nb_pipeline = make_pipeline(
    TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None),
    MultinomialNB(alpha=0.1)
)

# 训练模型
nb_pipeline.fit(X_train, y_train)

# 评估模型
y_pred_nb = nb_pipeline.predict(X_test)
print("测试集准确率: {:.4f}".format(accuracy_score(y_test, y_pred_nb)))
print("\n分类报告:")
print(classification_report(y_test, y_pred_nb))

# 5. 方法二：随机森林分类器
print("\n===== 随机森林分类器 =====")
# 创建管道：TF-IDF向量化 + 随机森林
rf_pipeline = make_pipeline(
    CountVectorizer(tokenizer=chinese_tokenizer, token_pattern=None, max_features=3000),
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
)

# 训练模型
rf_pipeline.fit(X_train, y_train)

# 评估模型
y_pred_rf = rf_pipeline.predict(X_test)
print("测试集准确率: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("\n分类报告:")
print(classification_report(y_test, y_pred_rf))


# 方法三：支持向量机
print("\n===== 支持向量机 =====")
svc_pipeline = make_pipeline(
    CountVectorizer(
        tokenizer=chinese_tokenizer,  # 使用jieba分词
        token_pattern=None,           # 禁用默认token模式
        max_features=5000,            # 限制特征数量
        ngram_range=(1, 2)            # 包含1-gram和2-gram
    ),
    SVC(
        kernel='linear',              # 线性核函数（适合文本数据）
        C=1.0,                        # 正则化参数
        probability=True,             # 启用概率预测
        random_state=42
    )
)
svc_pipeline.fit(X_train, y_train)
y_pred_svc = svc_pipeline.predict(X_test)
print("\nSVC模型评估结果:")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_svc):.4f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_svc))

# 6. 模型比较
print("\n===== 模型比较 =====")
print(f"朴素贝叶斯准确率: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"随机森林准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"支持向量机准确率: {accuracy_score(y_test, y_pred_svc):.4f}")

# 7. 预测新样本
test_queries = [
    "播放周杰伦的七里香",
    "查一下明天去上海的火车票",
    "我想看权力的游戏第三季",
    "帮我找找附近的美食",
    "明天北京的天气怎么样"
]

print("\n===== 新样本预测 =====")
for query in test_queries:
    tokenized_query = " ".join(jieba.lcut(query))

    # 使用两个模型分别预测
    nb_pred = nb_pipeline.predict([tokenized_query])[0]
    rf_pred = rf_pipeline.predict([tokenized_query])[0]
    svc_pred = svc_pipeline.predict([tokenized_query])[0]

    print(f"查询: '{query}'")
    print(f"  朴素贝叶斯预测: {nb_pred}")
    print(f"  随机森林预测: {rf_pred}")
    print(f"  支持向量机预测: {svc_pred}")
    print("-" * 50)

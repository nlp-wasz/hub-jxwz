import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: 读取数据集，并分割出训练集和测试集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print("数据集预览:")
print(dataset.head(5))

X = dataset[0] # 文本内容
y = dataset[1] # Query类别

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")


# Step 2: 提取文本特征
# -- 2.1 分词
X_train_cut = X_train.apply(lambda x: " ".join(jieba.lcut(x)))
X_test_cut = X_test.apply(lambda x: " ".join(jieba.lcut(x)))

# -- 2.2 使用CountVectorizer提取文本特征
vector = CountVectorizer()
X_train_vec = vector.fit_transform(X_train_cut) # .fit_transform() = .fit() + .transform()
X_test_vec = vector.transform(X_test_cut) #Note: 对于测试集，就不能再用fit了，而是应该直接使用训练集fit出来的特征空间
print(f"特征维度（词典大小）: {X_train_vec.shape[1]}")


# Step 3: 训练并测试一个KNN分类模型
# -- 3.1 初始化和训练
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train_vec, y_train)
print("-" * 30)
print("KNN模型训练完成。")

# -- 3.2 使用测试集测试KNN模型
y_pred_KNN = model_KNN.predict(X_test_vec)
accuracy_KNN = accuracy_score(y_test, y_pred_KNN)
print(f"KNN 模型在测试集上的准确率: {accuracy_KNN:.4f}")


# Step 4: 训练并测试一个Logistic Regression分类模型
# -- 4.1 初始化和训练
model_LR = LogisticRegression(max_iter=1000)
model_LR.fit(X_train_vec, y_train)
print("-" * 30)
print("LR模型训练完成。")

# -- 4.2: 使用测试集测试LR模型
y_pred_LR = model_LR.predict(X_test_vec)
accuracy_LR = accuracy_score(y_test, y_pred_LR)
print(f"逻辑回归模型在测试集上的准确率: {accuracy_LR:.4f}")

print("-" * 30)
print()
print()


# ====================================================================
# LAST STEP: 封装一个分类函数
# ====================================================================
def predict_category(text_input, vectorizer, classifier):
    text_cut = " ".join(jieba.lcut(text_input))
    text_vec = vectorizer.transform([text_cut])
    prediction = classifier.predict(text_vec)
    predicted_label = prediction[0]
    print("-" * 20)
    print(f"输入文本: '{text_input}'")
    print(f"预测类别: {predicted_label}")
    return predicted_label

sentence1 = "明天去北京的高铁票还有吗"
sentence2 = "播放一首周杰伦的歌"
sentence3 = "我想看一部吴京的动作电影"
print("--- 使用LR模型预测 ---")
predict_category(sentence1, vector, model_LR)
predict_category(sentence2, vector, model_LR)
predict_category(sentence3, vector, model_LR)
print("\n--- 使用KNN模型预测 ---")
predict_category(sentence1, vector, model_KNN)
predict_category(sentence2, vector, model_KNN)
predict_category(sentence3, vector, model_KNN)

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def testTFIModel(model, modelName):
    new_text = ["我喜欢唱周杰伦的歌"]
    new_text_cut = [" ".join(jieba.cut(new_text[0]))]
    new_vec = tfidf.transform(new_text_cut)
    print(f"模型{modelName} 在用TFI分词对{new_text}的预测结果:", model.predict(new_vec))

def testCountVectorizerModel(model, modelName):
    test_query = "边打电话边唱歌导航到德国"
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print(f"模型{modelName} 在用CountVectorizer对{test_query}的预测结果:", model.predict(test_feature))

data = pd.read_csv('./dataset.csv', header=None, sep='\t', names=['text', 'label'], encoding='utf-8')
data['text_cut'] = data['text'].apply(lambda x: " ".join(jieba.lcut(x)))


# CountVectorizer分词
print(data['text_cut'].values)
vector = CountVectorizer()
vector.fit(data['text_cut'])
input_feature = vector.transform(data['text_cut'])

# 逻辑回归
model = LogisticRegression()
model.fit(input_feature, data['label'])
testCountVectorizerModel(model, "LogisticRegression")

# KNN
model = KNeighborsClassifier()
model.fit(input_feature, data['label'])
testCountVectorizerModel(model, "KNeighborsClassifier")

# 贝叶斯
model = MultinomialNB()
model.fit(input_feature, data['label'])
testCountVectorizerModel(model, "MultinomialNB")

print("-------------------------------------------------------------------")

# 用TF提取分词
X_train, X_test, y_train, y_test = train_test_split(data['text_cut'], data['label'])
tfidf = TfidfVectorizer(max_features=15000)
X_train_tfidf = tfidf.fit_transform(X_train)

# 逻辑回归
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
testTFIModel(model, "LogisticRegression")


# KNN
model = KNeighborsClassifier()
model.fit(X_train_tfidf, y_train)
testTFIModel(model, "KNeighborsClassifier")

# 贝叶斯
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
testTFIModel(model, "MultinomialNB")

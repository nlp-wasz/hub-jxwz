import jieba
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

train_data = pd.read_csv('assets/dataset/作业数据-waimai_10k.csv')

cn_stopwords = pd.read_csv('assets/dataset/baidu_stopwords.txt', header=None)[0].values

print(train_data.head())
print(train_data['label'])

train_data['review'] = train_data['review'].apply(lambda x: " ".join([x for x in jieba.lcut(x) if x not in cn_stopwords]))

tfidf = TfidfVectorizer(ngram_range = (1,1) )

train_tfidf = tfidf.fit_transform(train_data['review'])

model = LinearSVC()
model.fit(train_tfidf, train_data['label'])

dump((tfidf, model), "assets/weights/tfidf_ml.pkl") # pickle 二进制
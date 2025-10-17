import pandas as pd
import jieba
from joblib import dump
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('assets/dataset/waimai_10k.csv', sep=',', header=None, skiprows=1)

cn_stopwords = pd.read_csv('assets/dataset/baidu_stopwords.txt', header=None)[0].values

train_data[1] = train_data[1].apply(lambda x: " ".join([x for x in jieba.lcut(x) if x not in cn_stopwords]))

tfidf = TfidfVectorizer(ngram_range = (1,1) )

train_tfidf = tfidf.fit_transform(train_data[1])

model = LinearSVC()
model.fit(train_tfidf, train_data[0])

dump((tfidf, model), "./assets/weights/tfidf_ml_waimai.pkl") # pickle 二进制


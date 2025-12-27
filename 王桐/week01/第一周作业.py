import jieba as jb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

#数据加载
data = pd.read_csv(r'E:\BaiduNetdiskDownload\第一周-课程介绍及大模型基础\课件\Week01\dataset.csv',encoding='UTF-8',sep='\t',header=None,names=['text','label'])
# print(data.head())
# print(data['label'].value_counts())
#定义函数切分文本
def word_split(text):
    return " ".join(jb.cut(text))

data['cute_text'] = data['text'].apply(word_split)

# 特征提取
tfidy=TfidfVectorizer(max_features=5000,ngram_range=(1,2))
x = tfidy.fit_transform(data['cute_text'])
y = data['label']

# 划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# 模型一：逻辑回归
print('-'*50,end='\n')
print('逻辑回归模型性能：')

model= LogisticRegression(max_iter=1000,class_weight='balanced') #模型初始化
model.fit(x_train,y_train)
y_re_model= model.predict(x_test)
print(classification_report(y_test,y_re_model))


# 模型二:决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print('-'*50,end='\n')
print('决策树模型性能:')

model = DecisionTreeClassifier(max_depth=50,min_samples_split=10,class_weight='balanced',random_state=42)
model.fit(x_train,y_train)
y_tr_model=model.predict(x_test)
print(classification_report(y_test,y_tr_model))
# 获取特征名和重要性
features = zip(tfidy.get_feature_names_out(), model.feature_importances_)
# 按重要性排序
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
# 打印Top3
print(f"特征重要性Top3: {sorted_features[:3]}")

# 模型对比
results={'LR':accuracy_score(y_test,y_re_model),
         'TR':accuracy_score(y_test,y_tr_model)}
print('-'*50,end='\n')
print("两种模型对比:")
for md,atc in sorted(results.items(),key=lambda x:x[1],reverse=True):
    print(f"{md:18s}:{atc:.4f}")

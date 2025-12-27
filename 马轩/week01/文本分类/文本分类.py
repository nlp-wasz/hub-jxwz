from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import tree
import jieba
import pandas as pd


dataset  = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(10))

input_sentence = dataset[0].apply(lambda x : " ".join(jieba.lcut(x)))
# print(input_sentence.head(5))
# print(input_sentence.values[0:5])
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)
# print(input_feature)

X,y = input_feature,dataset[1].values
train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)

def predict_statement(text,model,model_name):
    test_sentence  = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    print(f"带预测的文本:{text}")
    print(f"{model_name}预测结果:{model.predict(test_feature)}")
    print("\n")

print("方法一 KNN","=="*30)
model = KNeighborsClassifier()
model.fit(input_feature,dataset[1].values)
predict_statement("我想去火星旅游",model,"KNN")

print("方法二 决策树","=="*30)
model2  = tree.DecisionTreeClassifier()
model2.fit(train_x,train_y)
prediction2 = model2.predict(test_x)
print(f"决策树预测结果：{prediction2}")
print(f"决策树测试精度：{(test_y == prediction2).sum()/len(test_y)}")
predict_statement("飞虎队这部电影好看吗",model2,"决策树")

print("方法三 逻辑回归","=="*30)
model3  = linear_model.LogisticRegression(max_iter=1000)
model3.fit(train_x,train_y)
prediction3 = model3.predict(test_x)
print(f"逻辑回归预测结果：{prediction3}")
print(f"逻辑回归测试精度：{(test_y == prediction3).sum()/len(test_y)}")
predict_statement("抽空一起去黄山玩",model3,"逻辑回归")

# sklearn 代码
# 逻辑回归： 分类模型
# 加载给定的数据 sklearn

from sklearn import linear_model # 线性模型模块
from sklearn import tree # 决策树模型
from sklearn import datasets # 加载数据集
from sklearn.model_selection import train_test_split # 数据集划分
data = datasets.load_iris() # 植物分类的数据集
X, y = data.data, data.target

model = linear_model.LogisticRegression(max_iter=1000)

model.fit(X, y) # fit 就是训练模型

print(model)

train_x, test_x, train_y, test_y = train_test_split(X, y) # 数据划分 按一定的比例进行划分训练集和测试集
print(train_y)
print(test_y)

model = linear_model.LogisticRegression(max_iter=1000) # 初始化模型
model.fit(train_x, train_y)

prediction = model.predict(test_x)

print('预测结果：', prediction)
print('逻辑回归结果预测:', (test_y == prediction).sum(), len(test_x))


model = tree.DecisionTreeClassifier() # 初始化模型
model.fit(train_x, train_y)

prediction = model.predict(test_x)

print('预测结果：', prediction)
print('决策树结果预测:', (test_y == prediction).sum(), len(test_x))

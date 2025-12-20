
#导入所需要的函数包
#目标：将模型，套用在自己的数据集上
from sklearn import linear_model    #线性模块
from sklearn import tree    #新增决策树模块
from sklearn import datasets    #加载数据集
from sklearn.model_selection import train_test_split #数据集划分
from sklearn import neighbors

data = datasets.load_iris() #植物分类的数据集
# print(data)
X, y = data.data, data.target
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X, y) #fit 就是训练模型
print(model)

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=50)
# 训练集：调整模型的参数
# 测试集：验证模型的精度
print(train_y)
print("真实标签", test_y)

model = linear_model.LogisticRegression(max_iter=1000)  #模型初始化，人工设置了超参数， 从训练集学习到的叫模型参数
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("预测结果", prediction)
print("逻辑回归的测试结果", (test_y == prediction).sum(), len(test_x))

# 计算线性回归的准确率
print((test_y == prediction), len(test_x))
score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
print(f"线性回归模型准确率（使用 score 方法）：{score:.2f}")

model = tree.DecisionTreeClassifier()   #模型初始化
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("预测结果", prediction)
print("决策树的预测结果", (test_y == prediction).sum(), len(test_x))

# 计算逻辑树的准确率
print((test_y == prediction), len(test_x))
score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
print(f"决策树模型准确率（使用 score 方法）：{score:.2f}")

model = neighbors.KNeighborsClassifier(n_neighbors=1)  #模型初始化
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("预测结果", prediction)
print("KNN1的预测结果", (test_y == prediction).sum(), len(test_x))

# 计算KNN树的准确率
print((test_y == prediction), len(test_x))
score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
print(f"KNN-1模型准确率（使用 score 方法）：{score:.2f}")

model = neighbors.KNeighborsClassifier(n_neighbors=3)  #模型初始化
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("预测结果", prediction)
print("KNN-2的预测结果", (test_y == prediction).sum(), len(test_x))

# 计算KNN树的准确率
print((test_y == prediction), len(test_x))
score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
print(f"KNN-2模型准确率（使用 score 方法）：{score:.2f}")

model = neighbors.KNeighborsClassifier(n_neighbors=5)  #模型初始化
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("预测结果", prediction)
print("KNN-3的预测结果", (test_y == prediction).sum(), len(test_x))

# 计算KNN树的准确率
print((test_y == prediction), len(test_x))
score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
print(f"KNN-3模型准确率（使用 score 方法）：{score:.2f}")


# 搜索，遍历超参数,可以看多个参数的训练结果
for i in [1,3,5,7,9]:
    model = neighbors.KNeighborsClassifier(n_neighbors=i)  #模型初始化
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    print("预测结果", prediction)
    print(f"KNN-{i}的预测结果", (test_y == prediction).sum(), len(test_x))

    # 计算KNN树的准确率
    print((test_y == prediction), len(test_x))
    score = model.score(test_x, test_y) #通过score方法比较模型预测准确率
    print(f"KNN-{i}模型准确率（使用 score 方法）：{score:.2f}")

#coding = utf-8
#author = "zhouxu"

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])


print(sklearn.__version__)

# 用python的机器学习库
clf = Perceptron(fit_intercept=True,
				 max_iter=1000,
				 tol = None,
				 shuffle=True)
# 用python的数据集
clf.fit(X, y)

# weights assigned to the feature.
print(clf.coef_)

# 截距 Constrants in decision function
print(clf.intercept_)

# 画布大小
plt.figure(figsize=(10, 10))

# 标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("鸢尾花线性数据实例")

plt.scatter(data[:50, 0], data[:50, 1], c='b', label = 'Iris-setosa')
plt.scatter(data[50:100, 0], data[50:100, 1], c = 'orange', label = 'Iris-versicolor')

# 画感知器的线
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)

plt.legend() #显示图列
plt.grid(False)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# 花时间要了解下这个 sklearn 至少了解下这里面的perceptron各个参数意思。
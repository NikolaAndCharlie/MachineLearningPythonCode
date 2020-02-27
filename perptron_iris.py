#coding = utf-8

import pandas as pd
import numpy as np
from sklearn.datasets import  load_iris
import matplotlib.pyplot as plt

#加载数据，from iris

iris = load_iris()
# 数据采用iris.data, 列标签使用feature_names
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['label'] = iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
df.label.value_counts()

# 画出数据
#注意后面的label 并不是df数据里面的label column index
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal height')
plt.ylabel('sepal width')
plt.legend()


data = np.array(df.iloc[:100, [0, 1, -1]])
#索引所有行，但要求第0， 1， 最后列
print(df.iloc[:100, [0, 1, -1]])
#:-1意思是除去最后一个
x, y = data[:,:-1], data[:, -1]

#将 y 中的 0 变为-1
y = np.array([1 if i == 1 else -1 for i in y])

#perceptron
class Model:
	def __init__(self):
		self.w = np.ones(len(data[0]) -1, dtype = np.float32)
		self.b = 0
		self.l_rate = 0.1
		
	def sign(self, x, w, b):
		# 向量的点积
		# 函数间隔
		y = np.dot(x, w) + b
		return y
	#随机梯度下降法：
	
	def fit(self, X_train, Y_train):
		is_woring = False
		while not is_woring:
			wrong_count = 0
			for d in range(len(X_train)):
				x = X_train[d]
				y = Y_train[d]
				if y * self.sign(x, self.w, self.b) <= 0:
					self.w = self.w + self.l_rate * np.dot(y, x)
					self.b = self.b + self.l_rate * y
					is_woring = False
					wrong_count += 1
			if wrong_count == 0:
				is_woring = True
				print(self.w, self.b)
				return "Perceptron Model!"
	def score(self):
		pass

perceptron = Model()
perceptron.fit(x, y)

#draw the result
# linspace ,开始点，结束点，点数
x_points = np.linspace(4, 7, 10)
print(x_points)
# 不要局限的思考y_是什么，其实二维图像是x1, 和x2的几何关系
y_= -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
print(y_)
plt.plot(x_points, y_)
plt.show()
#author = zhouxu
#coding = 'utf-8'
# python 实现，遍历所有数据点，找出n个距离最近的点的分类情况，少数服从多数。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 加载数据，鸢尾花数据
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["label"] = iris.target
df.columns = ["Sepal length", "Sepal width", "petal length", "petal width", "label"]

plt.scatter(df[:50]["Sepal length"], df[:50]["Sepal width"], label= "0")
plt.scatter(df[50:100]["Sepal length"], df[50:100]["Sepal width"], label = "1")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal width")
plt.legend()


# 将数据放入一个数组中, 并分类。
data = np.array(df.iloc[:100, [0, 1, -1]])
x, y = data[:, :-1], data[:, -1]
# 将数据集分为训练集和验证集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
class KNN:
	def __init__(self, x_train, y_train, n_neightbors = 3, p = 2):
		"""
		:param x_train:
		:param y_train:
		:param n_neightbors:邻近点个数（K值）
		:param p: 度量方式
		"""
		self.n = n_neightbors
		self.p = p
		self.x_train = x_train
		self.y_train = y_train
	def predict(self, X):
		# 取出n个点
		knn_list = []
		for i in range(self.n):
			dist = np.linalg.norm(X - self.x_train[i], ord = self.p)
			knn_list.append((dist, self.y_train[i]))
		
		for i in range(self.n, len(self.x_train)):
			# 查询knn_list中第一维的最大值的index.
			max_index = knn_list.index(max(knn_list, key = lambda x: x[0]))
			# 求范式函数，也就是knn度量
			dist = np.linalg.norm(X - self.x_train[i], ord = self.p)
			# 这里其实是寻求最小的距离
			# 如果dist比knn_list现有的最大距离小，那么就记录，写入knn_list
			# 因为只用去更新最大的距离就行了，最小的距离没关系
			if knn_list[max_index][0] > dist:
				knn_list[max_index] = (dist, self.y_train[i])
		# 统计
		#统计分类最多的点
		knn = [k[-1] for k in knn_list]
		# 按照标签计数
		count_pairs = Counter(knn)
		max_count = sorted(count_pairs.items(), key = lambda x: x[1])[-1][0]
		return max_count
	#准确率：
	def score(self, x_test, y_test):
		right_count = 0
		for x, y in zip(x_test, y_test):
			label = self.predict(x)
			if label == y:
				right_count += 1
		print(right_count / len(x_test))
		return right_count / len(x_test)
# 初始化 训练集
clf = KNN(x_train, y_train)
# 测试机检测准确率
clf.score(x_test, y_test)

#预测一个点
test_point = [3, 4]
print("Test point: {}", format(clf.predict(test_point)))

plt.scatter(df[:50]["Sepal length"], df[0:50]["Sepal width"], label = "0")
plt.scatter(df[50:100]["Sepal length"], df[50:100]["Sepal width"], label = "1")
plt.plot(test_point[0], test_point[1], 'bo', label = "test point")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.show()
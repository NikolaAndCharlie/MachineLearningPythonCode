# author = "zhou xu"
# coding = utf-8


import os

# 实验数据集

traning_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]

w = [0, 0] #因为输入特征空间是二维的，算是二维向量
b = 0
h = 1;
#根据w, b梯度计算公式，按照一定的学习率更新w,b
#采用梯度下降法，（沿着梯度方向下降最快，沿着梯度反方向增加最快）
#这里假设学习率（步长）为1
def update(item):
	global w, b, h
	w[0] = w[0] + h * item[1] * item[0][0]
	w[1] = w[1] + h * item[1] * item[0][1]
	b = b + h * item[1]
	print(w, b)

#计算损失函数，也就是那个函数间隔，以区分是否出现误判点。
#y(wx + b) > 0的为误判点。

def cal(item):
	global  w, b
	function_distance = 0
	for i in range(len(item)):
		function_distance += item[0][i] * w[i]
	function_distance += b
	function_distance *= item[1]
	return function_distance

#迭代函数，知道不再出现误判点

def check():
	flag = False
	for item in traning_set:
		if cal(item) <= 0:
			flag = True
			update(item)
	if not flag:
		print("Result: w :"  + str(w) + " b: " + str(b))
		os._exit(0)
	flag = False
	
if __name__ == "__main__":
	for i in range(1000):
		check()
	
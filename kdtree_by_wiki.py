from collections import namedtuple
from operator import itemgetter
from pprint import pformat

class Node(namedtuple('Node', 'location left_child right_child')):
	def __repr__(self):
		return pformat(tuple(self))
	
def kdtree(point_list, depth: int = 0):
	if not point_list:
		return None
	
	k = len(point_list[0])
	axis = depth % k
	# sort(key = None, reverse = False)
	#reverse = False 降序， reverse = True 升序。
	#这里假定一开始一特征向量的第一维开始开始排序，方便下一步寻找中位数。
	#注意这里的depth，它在每次周期的时候都会更新到特征向量的下一维。
	#特征向量维度排序的跟踪是kdtree的周期循环。
	point_list.sort(key = itemgetter(axis))
	# 寻找当前axis维度的中位数的index。
	median = len(point_list) // 2
	
	# 开始递归，处理每个维度以中位数开始建立超平面分割（splitting plane）
	return Node(
		location = point_list[median],
		#注意这里有可能输入的是一个空的data set。由中位数的index而定。
  		left_child = kdtree(point_list[:median], depth + 1),
		right_child = kdtree(point_list[median + 1:], depth + 1)
	)

def dum(num):
	print(num)
	if num > 0:
		dum(num - 1)
	else:
		print("-----------")
	print(num)

def main():
	"""Example usage"""
	point_list = [(7, 2), ( 5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]#
	tree = kdtree(point_list)
	print(tree)
#	dum(3)

if __name__ == '__main__':
	main()
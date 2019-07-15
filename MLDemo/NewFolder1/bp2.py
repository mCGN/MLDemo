import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np

#输入是4维的向量（4），隐藏层(2)，输出是（3）
def sigmod(z):
	return 1 / (1 + np.exp(-z))

def dsigmod(a):
	return a * (1 - a)

def softmax(z):
	e = np.exp(z)
	return e / np.sum(e,axis=1,keepdims=True)

class BP(object):
	def __init__(self):
		iris = datasets.load_iris()
		self.LL = iris.data
		self.m,self.n = self.LL.shape
		self.x = np.array(self.LL)
		self.y = np.zeros((self.m,3))
		self.target = np.array(iris.target)
		for i in range(self.m):
			self.y[i][iris.target[i]] = 1

		self.w1 = np.zeros((4,3))
		self.w2 = np.zeros((3,3))
		self.b1 = np.zeros((1,3))
		self.b2 = np.zeros((1,3))

		self.rate = 0.5
	

	def backPropagation(self,sample,y,p=0):
		m,n = sample.shape
		a1 = sigmod(np.dot(sample,self.w1) + self.b1)
		a2 = softmax(a1.dot(self.w2) + self.b2)

		loss = np.sum(-np.log(a2[range(self.m),self.target])) / self.m
		
		l3 = a2
		l3[range(self.m),self.target]-=1
		l3/=self.m

		dw2 = np.dot(a2.T,l3)
		db2 = np.sum(l3,axis=0,keepdims=True)

		l2 = np.dot(l3,self.w2) * (dsigmod(a1))


		dw1 = np.dot(self.x.T,l2)
		db1 = np.sum(l2,axis=0,keepdims=True)

		self.w2 -= self.rate * dw2
		self.b2 -= self.rate * db2
		self.w1 -= self.rate * dw1
		self.b1 -= self.rate * db1

		if p == 0:
			print(loss)

	def predict(self,input):
		a1 = sigmod(input.dot(self.w1) + self.b1)
		a2 = softmax(a1.dot(self.w2) + self.b2)
		return a2

bp = BP()

for i in range(20000):
	bp.backPropagation(bp.x,bp.y,i%100)

print(bp.w1)
print(bp.b1)
print(bp.w2)
print(bp.b2)

print(bp.predict(bp.x[1]))
print(bp.y[1])
print(bp.predict(bp.x[60]))
print(bp.y[60])
print(bp.predict(bp.x[120]))
print(bp.y[120])



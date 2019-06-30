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
	return e / np.sum(e)

class BP(object):
	def __init__(self):
		iris = datasets.load_iris()
		self.LL = iris.data
		self.m,self.n = self.LL.shape
		self.x = np.array(self.LL)
		self.y = np.zeros((self.m,3))
		for i in range(self.m):
			self.y[i][iris.target[i]] = 1

		self.w1 = np.zeros((4,3))
		self.w2 = np.zeros((3,3))
		self.b1 = np.zeros((1,3))
		self.b2 = np.zeros((1,3))

		self.rate = 0.002
	

	def backPropagation(self,sample,y,p=1):
		m,n = sample.shape
		a1 = sigmod(sample.dot(self.w1) + self.b1)
		a2 = sigmod(a1.dot(self.w2) + self.b2)

		l2 = -(y - a2) * dsigmod(a2)
		l1 = l2.dot(self.w2.T) * dsigmod(a1)

		dw2 = a1.T.dot(l2)
		db2 = np.sum(l2,axis=0)

		dw1 = sample.T.dot(l1)
		db1 = np.sum(l1,axis=0)

		self.w2 -= self.rate * dw2
		self.w1 -= self.rate * dw1
		self.b2 -= self.rate * db2
		self.b1 -= self.rate * db1

		if p==0:
			print (np.sum(1/m*(y-a2)**2))

	def predict(self,input):
		a1 = sigmod(input.dot(self.w1) + self.b1)
		a2 = sigmod(a1.dot(self.w2) + self.b2)
		return a2

bp = BP()

for i in range(1500):
	for k in range(bp.m):
		bp.backPropagation(bp.x[k].reshape(1,bp.n),bp.y[k].reshape(1,3),k)

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



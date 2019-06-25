import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np

#输入是4维的向量（4），隐藏层(2)，输出是（3）
iris = datasets.load_iris()
LL = iris.data[0:1].T
n,m = LL.shape

x = np.array(LL) #n*4
y = np.zeros((3,m))
for i in range(m):
    y[iris.target[i]][i] = 1

W1 = np.ones((4,2))
W2 = np.ones((2,3))
b1 = np.ones((2,1))
b2 = np.ones((3,1))

def sigmod(z):
	return 1 / (1 + np.exp(-z))


def ds(o):
	return o * (1 - o)

rate = 0.01

for i in range(1000):
	a1 = sigmod(W1.T.dot(x) + b1)
	a2 = sigmod(W2.T.dot(a1) + b2)
	
	#[1,0,0] - [0.1,0.5,0.5] = 2(1-0.1)
	l3 = 2 * (y - a2) * a2 * (1 - a2)

	db2 = np.sum(l3,axis = 1).T
	dw2 = a1.dot(l3.T)

	l2 = W1 * l3 * a1 * (1 - a1)

print(predict(x))


def propagation():
	pass

def backPropagation():

	pass



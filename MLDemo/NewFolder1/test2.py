import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np

#输入是4维的向量（4），隐藏层(2)，输出是（3）
iris = datasets.load_iris()
LL = iris.data
n = len(LL)

x = np.array(LL) #n*4
y = np.zeros((n,3))
for i in range(len(iris.target)):
    y[i][iris.target[i]] = 1

W1 = np.ones((4,2))
W2 = np.ones((2,3))
b1 = np.ones((1,2))
b2 = np.ones((1,3))

def sigmod(z):
	return 1 / (1 + np.exp(-z))


def ds(o):
	return o*(1-o)

rate = 0.01

for i in range(1000):
	a2 = sigmod(x.dot(W1) + b1)
	a3 = sigmod(a2.dot(W2) + b2)
	
	#[1,0,0] - [0.1,0.5,0.5] = 2(1-0.1)
	2*(y-a2)



def predict(it):
	a1 = sigmod(it.dot(W1) + b1)
	a2 = sigmod(a1.dot(W2) + b2)
	return a2

print(predict(x))


def propagation():
	pass

def backPropagation():

	pass


